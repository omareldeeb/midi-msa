import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch

from .base_dataset import BaseMidiDataset
from .utils import widen_temporal_events, get_piano_roll_cache_path, get_sslm_cache_path, compute_sslms_from_midi_path, load_annotation
from .label_preprocessor import preprocess_labels


class TCNMidiDataset(BaseMidiDataset):
    def __init__(
        self,
        midi_dir: Union[str, Path],
        annotation_dir: Union[str, Path],
        midi_files: Optional[List[str]] = None,
        target_ticks_per_beat: int = 4,
        segment_function_vocab: Optional[List[str]] = None,
        label_map: Optional[Dict[str, str]] = None,
        compute_beats: bool = True,
        compute_downbeats: bool = True,
        compute_segment_labels: bool = True,
        instrument_overtones: bool = True,
        separate_drums: bool = True,
        use_sslms: bool = True,
        piano_roll_dir: Optional[Union[str, Path]] = None,
        sslm_dir: Optional[Union[str, Path]] = None,
        transpose_augmentation: bool = True,
        **kwargs
    ):
        super().__init__(
            target_ticks_per_beat=target_ticks_per_beat,
            instrument_overtones=instrument_overtones,
            separate_drums=separate_drums,
            transpose_augmentation=transpose_augmentation,
            use_sslms=use_sslms,
            compute_segment_labels=compute_segment_labels,
            segment_function_vocab=segment_function_vocab,
            label_map=label_map,
            midi_dir=midi_dir,
            annotation_dir=annotation_dir,
            midi_files=midi_files,
            piano_roll_dir=piano_roll_dir,
            sslm_dir=sslm_dir,
        )

        self.compute_beats = compute_beats
        self.compute_downbeats = compute_downbeats

        # Load measure data if available
        # try:
        #     with open("slms_metadata/measures_qn.json", "r") as f:
        #         self.measures_qn = json.load(f)
        # except FileNotFoundError:
        #     self.measures_qn = {}
        #     print("Warning: measures_qn.json not found, beat/downbeat computation will be limited")

    def __len__(self) -> int:
        return len(self.midi_file_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_id = self.midi_file_ids[idx]

        # Paths
        annotation_path = self.annotation_dir / f"{file_id}_functions_qn.json"

        # Check for cached piano roll
        cache_path = get_piano_roll_cache_path(file_id, self.piano_roll_dir, self.target_ticks_per_beat)
        if cache_path and cache_path.exists():
            # Load cached piano roll
            piano_roll_dict = torch.load(cache_path)

        else:
            # Compute piano roll
            piano_roll_dict = self._compute_piano_roll(file_id)
            if piano_roll_dict is None:
                raise ValueError(f'empty piano roll dict for {file_id}')
                return self._get_empty_sample()

            # Save to cache if caching is enabled
            if cache_path:
                torch.save(piano_roll_dict, cache_path)

        measure_ticks = piano_roll_dict['measure_ticks']
        time_signatures = piano_roll_dict['time_signatures']
        piano_roll = piano_roll_dict['piano_roll']

        if self.transpose_augmentation:
            piano_roll = self.apply_transpose_augmentation(piano_roll)

        # Cut silence from beginning
        # print(f"Piano roll shape before trimming: {piano_roll.shape}")
        # # piano_roll shape is (C, F, T); sum over channels and frequencies to get per-frame energy
        # print(piano_roll[:, :, 0])
        # non_silent_cols = torch.where(piano_roll.sum(dim=(0, 1)) > 0)[0]
        # print(non_silent_cols)
        # if len(non_silent_cols) == 0:
        #     return self._get_empty_sample()
        # first_non_silent = non_silent_cols[0].item()
        # # slice along time axis (last dim)
        # piano_roll = piano_roll[:, :, first_non_silent:]
        # print(f"First non-silent frame at {first_non_silent}")
        # print(f"Piano roll shape after trimming: {piano_roll.shape}")

        num_time_frames = piano_roll.shape[-1]
        
        # Create sample dict
        sample = {}
        if self.use_sslms:
            sslm_cache_path = get_sslm_cache_path(file_id, self.sslm_dir, self.target_ticks_per_beat)
            if sslm_cache_path and sslm_cache_path.exists():
                sslm_data = torch.load(sslm_cache_path)
                sslm_near = sslm_data["sslm_near"]
                sslm_far = sslm_data["sslm_far"]
            else:
                # Merge piano roll across channels for SSLM computation by summing
                midi_path = self.midi_dir / f"{file_id[0]}" / f"{file_id}.mid"
                sslm_near, sslm_far = compute_sslms_from_midi_path(p=midi_path,
                                                                   target_ticks_per_beat=self.target_ticks_per_beat)
                if sslm_cache_path:
                    torch.save({"sslm_near": sslm_near, "sslm_far": sslm_far}, sslm_cache_path)
            
            # Match dims to piano roll
            height = piano_roll.shape[-2]
            sslm_near = sslm_near[:height, :num_time_frames]
            sslm_far = sslm_far[:height, :num_time_frames]
            # add channel dim
            sslm_near = sslm_near.unsqueeze(0)
            sslm_far = sslm_far.unsqueeze(0)
            # pad height if needed
            if sslm_near.shape[-2] < height:
                pad_amount = height - sslm_near.shape[-2]
                sslm_near = torch.nn.functional.pad(sslm_near, (0, 0, 0, pad_amount))
            if sslm_far.shape[-2] < height:
                pad_amount = height - sslm_far.shape[-2]
                sslm_far = torch.nn.functional.pad(sslm_far, (0, 0, 0, pad_amount))

            sample["sslm_near"] = sslm_near
            sample["sslm_far"] = sslm_far

        # compute_piano_roll now always returns 3 channels: non-drums, overtones, drums
        # consolidate output format here
        if not self.separate_drums and self.instrument_overtones:
            piano_roll = torch.stack([
                    piano_roll[0] + piano_roll[2],
                    piano_roll[1],
                    torch.zeros_like(piano_roll[0])
            ])
        elif self.separate_drums and not self.instrument_overtones:
            piano_roll = torch.stack([
                piano_roll[0],
                torch.zeros_like(piano_roll[0]),
                piano_roll[2]
            ])
        elif not self.separate_drums and not self.instrument_overtones:
            piano_roll = torch.stack([
                piano_roll[0] + piano_roll[2],
                torch.zeros_like(piano_roll[0]),
                torch.zeros_like(piano_roll[0])
            ])
        
        piano_roll = torch.clip(piano_roll, 0.0, 1.0)
        sample["piano_roll"] = piano_roll

        # Add measure ticks
        sample["measure_ticks"] = torch.tensor(measure_ticks, dtype=torch.long)
        
        # Load annotations
        annotations = load_annotation(annotation_path)
        annotations = preprocess_labels(annotations, label_map=self.label_map)

        segment_qns = [ann[0] for ann in annotations]
        segment_labels = [ann[1] for ann in annotations]
        if segment_qns and segment_qns[0] != 0:
            segment_qns = [0] + segment_qns
            segment_labels = ['Start'] + segment_labels

        # Adjust for target ticks per beat
        segment_ticks = [int(round(qn * self.target_ticks_per_beat)) for qn in segment_qns]

        # Crop annotations to piano roll window
        activation_segment_ticks = []
        activation_segment_labels = []
        for tick, label in zip(segment_ticks, segment_labels):
            if tick < num_time_frames:
                activation_segment_ticks.append(tick)
                activation_segment_labels.append(label)
        sample['segment_ticks_in_piano_roll'] = activation_segment_ticks
        sample['segment_labels_in_piano_roll'] = activation_segment_labels

        # Compute segment boundary activation
        if self.compute_segment_labels:
            segment_activation = torch.zeros(num_time_frames, dtype=torch.float32)
            segment_activation[activation_segment_ticks] = 1.0
            segment_activation = widen_temporal_events(segment_activation, num_neighbors=2)
            sample["segment_activation"] = segment_activation

            # Create segment label activations (for each frame, which segment it belongs to)
            segment_label_activations = self._create_segment_label_activations(
                activation_segment_ticks, activation_segment_labels, num_time_frames
            )
            sample["segment_label_activations"] = segment_label_activations
        
        # Compute beat and downbeat activations from measures
        if (self.compute_beats or self.compute_downbeats) and measure_ticks:
            if self.compute_beats:

                # Use actual time signature from MIDI file to add beats between measures
                beat_ticks = []
                for i in range(len(measure_ticks) - 1):
                    start_tick = measure_ticks[i]
                    end_tick = measure_ticks[i + 1]
                    
                    # Find the applicable time signature for this measure
                    # Use the most recent time signature before or at the start of this measure
                    applicable_time_sig = time_signatures[0]  # Default to first (or default 4/4)
                    for tick_pos, numerator, denominator in time_signatures:
                        if tick_pos <= start_tick:
                            applicable_time_sig = (tick_pos, numerator, denominator)
                        else:
                            break
                    
                    _, num_beats, denominator = applicable_time_sig
                    # if denominator != 4:
                        # print(f"Warning: time signature denominator {denominator} not 4 for {midi_path}")
                    beat_interval = (end_tick - start_tick) / num_beats
                    for j in range(num_beats):
                        beat_ticks.append(start_tick + int(round(j * beat_interval)))
                    beat_ticks = [tick for tick in beat_ticks if tick < num_time_frames]
                
                beat_activation = torch.zeros(num_time_frames, dtype=torch.float32)
                beat_activation[beat_ticks] = 1.0
                beat_activation = widen_temporal_events(beat_activation, num_neighbors=1)
                sample["beat_activation"] = torch.tensor(beat_activation, dtype=torch.float32)
            
            if self.compute_downbeats:
                downbeat_activation = torch.zeros(num_time_frames, dtype=torch.float32)
                downbeat_measure_ticks = [x for x in measure_ticks if x < num_time_frames]
                downbeat_activation[downbeat_measure_ticks] = 1.0
                downbeat_activation = widen_temporal_events(downbeat_activation, num_neighbors=1)
                sample["downbeat_activation"] = downbeat_activation

        sample['file_id'] = file_id
        return sample
    
    def _create_segment_label_activations(
        self, 
        segment_ticks: List[int], 
        segment_labels: List[str], 
        num_time_frames: int
    ) -> torch.Tensor:
        """Create frame-wise segment function labels."""
        segment_label_activations = torch.zeros(num_time_frames, dtype=torch.long)

        assert len(segment_ticks) == len(segment_labels), "segment_ticks and segment_labels have unequal sizes"

        # Process segments
        for i in range(len(segment_labels)):
            start_tick = segment_ticks[i]
            end_tick = segment_ticks[i + 1] if i < len(segment_labels) - 1 else num_time_frames
            
            # Extract function and find its index in vocabulary
            # function = segment_labels[i].split(";")[0].strip()
            function = segment_labels[i]
            if function in self.segment_function_vocab:
                class_idx = self.segment_function_vocab.index(function)
                segment_label_activations[start_tick:end_tick] = class_idx
            else:
                print(f"Warning: Function '{function}' not in vocabulary, skipping")
        
        return segment_label_activations
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return an empty sample when no valid piano rolls are found."""
        sample = {
            "piano_roll": torch.zeros(3, 128, 1),
        }
        if self.compute_segment_labels:
            sample["segment_activation"] = torch.zeros(1)
            sample["segment_label_activations"] = torch.zeros(1, dtype=torch.long)
        if self.compute_beats:
            sample["beat_activation"] = torch.zeros(1)
        if self.compute_downbeats:
            sample["downbeat_activation"] = torch.zeros(1)
        if self.use_sslms:
            sample["sslm_near"] = torch.zeros(1, 128, 1)
            sample["sslm_far"] = torch.zeros(1, 128, 1)
        return sample


# class TCNAudioDataset(Dataset):
#     def __init__(
#         self,
#         audio_dir: Union[str, Path],
#         annotation_dir: Union[str, Path],
#         audio_files: Optional[List[str]] = None,
#         sample_rate: int = 44100,
#         hop_length: int = 441,  # ~10ms hop at 44.1kHz
#         n_fft: int = 2048,
#         n_bands: int = 24,
#         fmin: float = 30.0,
#         fmax: Optional[float] = None,
#         segment_function_vocab: Optional[List[str]] = None,
#         compute_beats: bool = True,
#         compute_downbeats: bool = True,
#         compute_segments: bool = True,
#         **kwargs
#     ):
#         """
#         TCN Dataset for audio files with functional segment annotations.
        
#         Args:
#             audio_dir: Directory containing audio files
#             annotation_dir: Directory containing annotation files (*_labels_coarse_qn.json)
#             audio_files: Optional list of specific audio file IDs to use
#             sample_rate: Target sample rate for audio
#             hop_length: Hop length for spectrogram computation
#             n_fft: FFT size for spectrogram computation
#             n_bands: Number of mel bands (or frequency bins for log filterbank)
#             fmin: Minimum frequency for filterbank
#             fmax: Maximum frequency for filterbank (None = Nyquist)
#             segment_function_vocab: List of segment function labels to classify
#             compute_beats: Whether to compute beat activations
#             compute_downbeats: Whether to compute downbeat activations
#             compute_segments: Whether to compute segment boundary activations
#         """
#         self.audio_dir = Path(audio_dir)
#         self.annotation_dir = Path(annotation_dir)
#         self.sample_rate = sample_rate
#         self.hop_length = hop_length
#         self.n_fft = n_fft
#         self.n_bands = n_bands
#         self.fmin = fmin
#         self.fmax = fmax or sample_rate // 2
#         self.compute_beats = compute_beats
#         self.compute_downbeats = compute_downbeats
#         self.compute_segments = compute_segments
        
#         # Get all annotation files and extract audio file IDs
#         if audio_files is None:
#             annotation_files = glob.glob(str(self.annotation_dir / "*_labels_coarse_qn.json"))
#             self.audio_file_ids = [Path(f).stem.replace("_labels_coarse_qn", "") for f in annotation_files]
#         else:
#             self.audio_file_ids = audio_files
            
#         # Filter out files that don't have both audio and annotation
#         valid_file_ids = []
#         for file_id in tqdm(self.audio_file_ids, desc="Validating audio files"):
#             # Try different audio extensions
#             audio_path = None
#             for ext in ['.wav', '.mp3', '.flac', '.m4a']:
#                 candidate_path = self.audio_dir / f"{file_id[0]}" / f"{file_id}{ext}"
#                 if candidate_path.exists():
#                     audio_path = candidate_path
#                     break
                    
#             annotation_path = self.annotation_dir / f"{file_id}_labels_coarse_qn.json"
            
#             if audio_path and annotation_path.exists():
#                 valid_file_ids.append((file_id, audio_path))
                
#         self.audio_file_ids = valid_file_ids
#         print(f"Found {len(self.audio_file_ids)} valid audio files with annotations")
        
#         # Build segment function vocabulary if not provided
#         if segment_function_vocab is None and compute_segments:
#             self.segment_function_vocab = self._build_segment_vocab()
#         else:
#             self.segment_function_vocab = segment_function_vocab or []
            
#         # Initialize madmom processors
#         self._init_madmom_processors()
        
#         # Load measure data if available
#         try:
#             with open("slms_metadata/measures_qn.json", "r") as f:
#                 self.measures_qn = json.load(f)
#         except FileNotFoundError:
#             self.measures_qn = {}
#             print("Warning: measures_qn.json not found, beat/downbeat computation will be limited")
    
#     def _init_madmom_processors(self):
#         """Initialize madmom audio processors."""
#         # Spectrogram processor
#         self.spec_processor = madmom.audio.spectrogram.LogarithmicFilteredSpectrogramProcessor(
#             num_bands=self.n_bands,
#             fmin=self.fmin,
#             fmax=self.fmax,
#             sample_rate=self.sample_rate,
#             hop_length=self.hop_length,
#             num_fft_bins=self.n_fft
#         )
        
#         # Beat tracking processors
#         if self.compute_beats or self.compute_downbeats:
#             self.beat_processor = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
#             self.beat_activation_processor = madmom.features.beats.RNNBeatProcessor()
            
#         # Downbeat tracking processors  
#         if self.compute_downbeats:
#             self.downbeat_processor = madmom.features.downbeats.DBNDownBeatTrackingProcessor(fps=100)
#             self.downbeat_activation_processor = madmom.features.downbeats.RNNDownBeatProcessor()
    
#     def _build_segment_vocab(self) -> List[str]:
#         """Build vocabulary of unique segment functions from all annotations."""
#         vocab = set()
        
#         for file_id, _ in tqdm(self.audio_file_ids, desc="Building segment vocabulary"):
#             annotation_path = self.annotation_dir / f"{file_id}_labels_coarse_qn.json"
#             with open(annotation_path, "r") as f:
#                 annotations = json.load(f)
                
#             for _, label in annotations:
#                 # Extract first function from semicolon-separated string
#                 if label != "End":
#                     function = label.split(";")[0].strip()
#                     vocab.add(function)
        
#         vocab = sorted(list(vocab))
#         print(f"Built segment vocabulary with {len(vocab)} unique functions: {vocab}")
#         return vocab
    
#     def __len__(self) -> int:
#         return len(self.audio_file_ids)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         file_id, audio_path = self.audio_file_ids[idx]
        
#         # Paths
#         annotation_path = self.annotation_dir / f"{file_id}_labels_coarse_qn.json"
        
#         # Load and process audio to log magnitude spectrogram
#         try:
#             # Load audio
#             signal = madmom.audio.signal.Signal(
#                 str(audio_path), 
#                 sample_rate=self.sample_rate,
#                 num_channels=1  # Convert to mono
#             )
            
#             # Compute log magnitude spectrogram
#             spectrogram = self.spec_processor(signal)
            
#             # Transpose to have frequency bins as channels (like piano roll)
#             # Shape: (n_bands, time_frames)
#             spectrogram = spectrogram.T
            
#             # Add batch dimension to match piano roll format
#             # Shape: (1, n_bands, time_frames) 
#             spectrogram = np.expand_dims(spectrogram, axis=0)
            
#             # Convert to tensor and normalize
#             spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
            
#             # Normalize to [0, 1] range
#             spec_min = spectrogram.min()
#             spec_max = spectrogram.max()
#             if spec_max > spec_min:
#                 spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
            
#         except Exception as e:
#             print(f"Error processing audio file {audio_path}: {e}")
#             return self._get_empty_sample()
        
#         num_time_frames = spectrogram.shape[-1]
#         fps = self.sample_rate / self.hop_length  # Frames per second
        
#         # Create sample dict
#         sample = {"spectrogram": spectrogram}
        
#         # Load annotations
#         with open(annotation_path, "r") as f:
#             annotations = json.load(f)
        
#         # Convert quarter note positions to time positions
#         # Assuming annotations are in quarter notes and we need to convert to seconds
#         # This conversion depends on the tempo, which we might need from MIDI
#         # For now, we'll use a default tempo of 120 BPM (2 beats per second)
#         tempo_bpm = 120  # Default tempo
#         beats_per_second = tempo_bpm / 60
#         seconds_per_quarter = 1 / beats_per_second
        
#         segment_times = [ann[0] * seconds_per_quarter for ann in annotations]
#         segment_labels = [ann[1] for ann in annotations]
        
#         # Convert time positions to frame indices
#         segment_frames = [int(round(t * fps)) for t in segment_times]
        
#         # Compute segment boundary activation
#         if self.compute_segments:
#             # Remove "End" markers for boundary detection
#             boundary_times = [t for t, label in zip(segment_times, segment_labels) if label != "End"]
#             segment_activation = create_target_activation(
#                 boundary_times,
#                 int(fps),
#                 num_time_frames
#             )
#             segment_activation = widen_temporal_events(segment_activation, num_neighbors=2)
#             sample["segment_activation"] = torch.tensor(segment_activation, dtype=torch.float32)
            
#             # Create segment label activations
#             segment_label_activations = self._create_segment_label_activations(
#                 segment_frames, segment_labels, num_time_frames
#             )
#             sample["segment_label_activations"] = segment_label_activations
        
#         # Compute beat and downbeat activations using madmom
#         if self.compute_beats:
#             try:
#                 # Get beat activation function
#                 beat_activation_fn = self.beat_activation_processor(signal)
#                 # Track beats
#                 beats = self.beat_processor(beat_activation_fn)
                
#                 # Convert beat times to frames
#                 beat_frames = [int(round(b * fps)) for b in beats]
#                 beat_frames = [f for f in beat_frames if 0 <= f < num_time_frames]
                
#                 beat_activation = torch.zeros(num_time_frames)
#                 beat_activation[beat_frames] = 1.0
#                 beat_activation = widen_temporal_events(beat_activation.numpy(), num_neighbors=1)
#                 sample["beat_activation"] = torch.tensor(beat_activation, dtype=torch.float32)
#             except Exception as e:
#                 print(f"Error computing beats for {audio_path}: {e}")
#                 sample["beat_activation"] = torch.zeros(num_time_frames)
        
#         if self.compute_downbeats:
#             try:
#                 # Get downbeat activation function
#                 downbeat_activation_fn = self.downbeat_activation_processor(signal)
#                 # Track downbeats
#                 downbeats = self.downbeat_processor(downbeat_activation_fn)
                
#                 # Extract just the downbeat times (first column)
#                 downbeat_times = downbeats[:, 0]
                
#                 # Convert downbeat times to frames
#                 downbeat_frames = [int(round(d * fps)) for d in downbeat_times]
#                 downbeat_frames = [f for f in downbeat_frames if 0 <= f < num_time_frames]
                
#                 downbeat_activation = torch.zeros(num_time_frames)
#                 downbeat_activation[downbeat_frames] = 1.0
#                 downbeat_activation = widen_temporal_events(downbeat_activation.numpy(), num_neighbors=1)
#                 sample["downbeat_activation"] = torch.tensor(downbeat_activation, dtype=torch.float32)
#             except Exception as e:
#                 print(f"Error computing downbeats for {audio_path}: {e}")
#                 sample["downbeat_activation"] = torch.zeros(num_time_frames)
        
#         return sample
    
#     def _create_segment_label_activations(
#         self, 
#         segment_frames: List[int], 
#         segment_labels: List[str], 
#         num_time_frames: int
#     ) -> torch.Tensor:
#         """Create frame-wise segment function labels."""
#         segment_label_activations = torch.zeros(num_time_frames, dtype=torch.long)
        
#         # Process segments
#         for i in range(len(segment_labels) - 1):
#             if segment_labels[i] == "End":
#                 continue
                
#             start_frame = segment_frames[i]
#             end_frame = segment_frames[i + 1]
            
#             # Clip to valid range
#             start_frame = max(0, start_frame)
#             end_frame = min(num_time_frames, end_frame)
            
#             # Extract function and find its index in vocabulary
#             function = segment_labels[i].split(";")[0].strip()
#             if function in self.segment_function_vocab:
#                 class_idx = self.segment_function_vocab.index(function)
#                 segment_label_activations[start_frame:end_frame] = class_idx
        
#         return segment_label_activations
    
#     def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
#         """Return an empty sample when audio processing fails."""
#         sample = {
#             "spectrogram": torch.zeros(1, self.n_bands, 1),
#         }
#         if self.compute_segments:
#             sample["segment_activation"] = torch.zeros(1)
#             sample["segment_label_activations"] = torch.zeros(1, dtype=torch.long)
#         if self.compute_beats:
#             sample["beat_activation"] = torch.zeros(1)
#         if self.compute_downbeats:
#             sample["downbeat_activation"] = torch.zeros(1)
#         return sample