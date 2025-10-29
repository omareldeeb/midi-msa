import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import glob

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
# import madmom

from .utils import parse_midi, create_piano_roll, parse_markers, create_target_activation, widen_temporal_events, compute_sslm
from .label_preprocessor import preprocess_labels


def transpose_augmentation(piano_roll, transpose_range=6):
    transpose_amount = torch.randint(-transpose_range, transpose_range, ())
    return torch.roll(piano_roll, transpose_amount.item(), dims=-2) # type: ignore


class TCNMidiDataset(Dataset):
    def __init__(
        self,
        midi_dir: Union[str, Path],
        annotation_dir: Union[str, Path],
        midi_files: Optional[List[str]] = None,
        target_ticks_per_beat: int = 4,
        segment_function_vocab: Optional[List[str]] = None,
        compute_beats: bool = True,
        compute_downbeats: bool = True,
        compute_segments: bool = True,
        instrument_overtones: bool = True,
        separate_drums: bool = True,
        compute_sslms: bool = False,
        piano_roll_dir: Optional[Union[str, Path]] = None,
        sslms_dir: Optional[Union[str, Path]] = None,
        transpose_augmentation: bool = True,
        **kwargs
    ):
        """
        TCN Dataset for MIDI files with functional segment annotations.

        Args:
            midi_dir: Directory containing MIDI files
            annotation_dir: Directory containing annotation files (*_labels_coarse_qn.json)
            midi_files: Optional list of specific MIDI file IDs to use
            target_ticks_per_beat: Target resolution for piano rolls
            segment_function_vocab: List of segment function labels to classify
            compute_beats: Whether to compute beat activations
            compute_downbeats: Whether to compute downbeat activations
            compute_segments: Whether to compute segment boundary activations
            instrument_overtones: Whether to include instrument overtones in piano roll
            separate_drums: Whether to separate drum tracks in piano roll
            piano_roll_dir: Optional directory to cache computed piano rolls
        """
        self.midi_dir = Path(midi_dir)
        self.annotation_dir = Path(annotation_dir)
        self.target_ticks_per_beat = target_ticks_per_beat
        self.compute_beats = compute_beats
        self.compute_downbeats = compute_downbeats
        self.compute_segments = compute_segments
        self.instrument_overtones = instrument_overtones
        self.separate_drums = separate_drums
        self.compute_sslms = compute_sslms
        self.piano_roll_dir = Path(piano_roll_dir) if piano_roll_dir else None
        self.sslms_dir = Path(sslms_dir) if sslms_dir else None
        self.transpose_augmentation = transpose_augmentation

        # Create cache directory if it doesn't exist
        if self.piano_roll_dir:
            self.piano_roll_dir.mkdir(parents=True, exist_ok=True)

        if self.sslms_dir:
            self.sslms_dir.mkdir(parents=True, exist_ok=True)

        # Get all annotation files and extract MIDI file IDs
        if midi_files is None:
            annotation_files = glob.glob(str(self.annotation_dir / "*_labels_coarse_qn.json"))
            self.midi_file_ids = [Path(f).stem.replace("_labels_coarse_qn", "") for f in annotation_files]
        else:
            self.midi_file_ids = midi_files
            
        # Filter out files that don't have both MIDI and annotation
        valid_file_ids = []
        for file_id in tqdm(self.midi_file_ids, desc="Validating files"):
            midi_path = self.midi_dir / f"{file_id[0]}" / f"{file_id}.mid"
            annotation_path = self.annotation_dir / f"{file_id}_labels_coarse_qn.json"

            if midi_path.exists() and annotation_path.exists():
                valid_file_ids.append(file_id)
                
        self.midi_file_ids = valid_file_ids
        print(f"Found {len(self.midi_file_ids)} valid MIDI files with annotations")
        
        # Build segment function vocabulary if not provided
        if segment_function_vocab is None and compute_segments:
            self.segment_function_vocab = self._build_segment_vocab()
        else:
            self.segment_function_vocab = segment_function_vocab
            
        # Load measure data if available
        try:
            with open("slms_metadata/measures_qn.json", "r") as f:
                self.measures_qn = json.load(f)
        except FileNotFoundError:
            self.measures_qn = {}
            print("Warning: measures_qn.json not found, beat/downbeat computation will be limited")

        # Precompute piano roll cache if caching is enabled
        if self.piano_roll_dir:
            self._precompute_piano_roll_cache()

    def _get_piano_roll_cache_path(self, file_id: str) -> Optional[Path]:
        """Get the cache path for a piano roll file."""
        if not self.piano_roll_dir:
            return None

        # Create a unique filename based on file_id and piano roll parameters
        cache_filename = (
            f"{file_id}_tpb{self.target_ticks_per_beat}"
            f"_ot{int(self.instrument_overtones)}"
            f"_sd{int(self.separate_drums)}.pt"
        )
        return self.piano_roll_dir / cache_filename
    
    def _get_sslm_cache_path(self, file_id: str) -> Optional[Path]:
        """Get the cache path for an SSLM file."""
        if not self.sslms_dir:
            return None

        cache_filename = f"{file_id}_sslm_tp{self.target_ticks_per_beat}.pt"
        return self.sslms_dir / cache_filename

    def _compute_piano_roll(self, file_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compute piano roll and adjusted time signatures for a given file.

        Returns:
            Dictionary containing 'piano_roll' and 'time_signatures', or None if computation fails
        """
        midi_path = self.midi_dir / f"{file_id[0]}" / f"{file_id}.mid"

        try:
            # Parse MIDI file
            track_data, ticks_per_beat, time_signatures = parse_midi(midi_path)

            # Create piano rolls for each track and merge them
            piano_rolls = []
            for track_name, note_data in track_data.items():
                piano_roll = create_piano_roll(
                    note_data,
                    ticks_per_beat,
                    chroma=False,
                    target_ticks_per_beat=self.target_ticks_per_beat,
                    instrument_overtones=self.instrument_overtones,
                    separate_drums=self.separate_drums
                )
                if piano_roll is not None:
                    piano_rolls.append(piano_roll)

            if len(piano_rolls) == 0:
                print(f"Warning: No valid piano rolls for {midi_path}")
                return None

            # Stack and merge piano rolls
            actual_length = max(pr.shape[-1] for pr in piano_rolls)
            for i, pr in enumerate(piano_rolls):
                piano_rolls[i] = torch.nn.functional.pad(
                    torch.tensor(pr),
                    (0, actual_length - pr.shape[-1])
                )

            piano_roll = torch.stack(piano_rolls).sum(dim=0).clamp(0, 127)
            piano_roll = piano_roll.float() / 127.0  # Normalize to [0, 1]

            # Adjust time signature tick positions for target ticks per beat
            adjusted_time_signatures = [
                (int(round(tick_pos * self.target_ticks_per_beat / ticks_per_beat)), numerator, denominator)
                for tick_pos, numerator, denominator in time_signatures
            ]

            return {
                "piano_roll": piano_roll,
                "time_signatures": adjusted_time_signatures
            }

        except Exception as e:
            print(f"Error computing piano roll for {file_id}: {e}")
            return None

    def _precompute_piano_roll_cache(self):
        """Precompute and cache all piano rolls."""
        if not self.piano_roll_dir:
            return

        # Check which files need to be cached
        files_to_cache = []
        for file_id in self.midi_file_ids:
            cache_path = self._get_piano_roll_cache_path(file_id)
            if cache_path and not cache_path.exists():
                files_to_cache.append(file_id)

        if not files_to_cache:
            print(f"All {len(self.midi_file_ids)} piano rolls already cached")
            return

        print(f"Caching {len(files_to_cache)} piano rolls...")
        for file_id in tqdm(files_to_cache, desc="Caching piano rolls"):
            cache_path = self._get_piano_roll_cache_path(file_id)
            if not cache_path or cache_path.exists():
                continue

            # Compute piano roll
            result = self._compute_piano_roll(file_id)
            if result is None:
                continue

            # Save to cache
            torch.save(result, cache_path)

    def _build_segment_vocab(self) -> List[str]:
        """Build vocabulary of unique segment functions from all annotations."""
        vocab = set()
        
        for file_id in tqdm(self.midi_file_ids, desc="Building segment vocabulary"):
            annotation_path = self.annotation_dir / f"{file_id}_labels_coarse_qn.json"
            with open(annotation_path, "r") as f:
                annotations = json.load(f)
                
            for _, label in annotations:
                # Extract first function from semicolon-separated string
                if label != "End":
                    function = label.split(";")[0].strip()
                    vocab.add(function)
        
        vocab = sorted(list(vocab))
        print(f"Built segment vocabulary with {len(vocab)} unique functions: {vocab}")
        return vocab
    
    def __len__(self) -> int:
        return len(self.midi_file_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_id = self.midi_file_ids[idx]

        # Paths
        annotation_path = self.annotation_dir / f"{file_id}_labels_coarse_qn.json"

        # Check for cached piano roll
        cache_path = self._get_piano_roll_cache_path(file_id)
        if cache_path and cache_path.exists():
            # Load cached piano roll
            cached_data = torch.load(cache_path)
            piano_roll = cached_data["piano_roll"]
            time_signatures = cached_data["time_signatures"]
        else:
            # Compute piano roll
            result = self._compute_piano_roll(file_id)
            if result is None:
                return self._get_empty_sample()

            piano_roll = result["piano_roll"]
            time_signatures = result["time_signatures"]

            # Save to cache if caching is enabled
            if cache_path:
                torch.save(result, cache_path)

        if self.transpose_augmentation:
            piano_roll = transpose_augmentation(piano_roll)

        num_time_frames = piano_roll.shape[-1]
        
        # Create sample dict
        sample = {"piano_roll": piano_roll}

        if self.compute_sslms:
            sslm_cache_path = self._get_sslm_cache_path(file_id)
            if sslm_cache_path and sslm_cache_path.exists():
                sslm_data = torch.load(sslm_cache_path)
                sslm_near = sslm_data["sslm_near"]
                sslm_far = sslm_data["sslm_far"]
            else:
                sslm_near = compute_sslm(piano_roll, L=int((14 / 0.5) * self.target_ticks_per_beat)) # 14s at 0.5 seconds per beat (120 BPM) at target resolution
                sslm_far = compute_sslm(piano_roll, L=int((88 / 0.5) * self.target_ticks_per_beat)) # 88s at 0.5 seconds per beat (120 BPM) at target resolution
                if sslm_cache_path:
                    torch.save({"sslm_near": sslm_near, "sslm_far": sslm_far}, sslm_cache_path)
            
            # Match dims to piano roll
            height = piano_roll.shape[-2]
            sslm_near = sslm_near[:height, :num_time_frames]
            sslm_far = sslm_far[:height, :num_time_frames]

            sample["sslm_near"] = sslm_near
            sample["sslm_far"] = sslm_far


        # Add measure ticks
        measure_ticks = None
        if file_id in self.measures_qn:
            measure_qns = self.measures_qn[file_id]
            measure_ticks = [int(round(qn * self.target_ticks_per_beat)) for qn in measure_qns]
            measure_ticks = [min(tick, num_time_frames - 1) for tick in measure_ticks]

            sample["measure_ticks"] = torch.tensor(measure_ticks, dtype=torch.long)
        
        # Load annotations
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
            annotations = preprocess_labels(annotations)

        # Convert quarter note positions to tick positions
        segment_qns = [ann[0] for ann in annotations]
        segment_labels = [ann[1] for ann in annotations]
        
        # Adjust for target ticks per beat
        segment_ticks = [int(round(qn * self.target_ticks_per_beat)) for qn in segment_qns]
        segment_ticks = [min(tick, num_time_frames - 1) for tick in segment_ticks]
        
        # Compute segment boundary activation
        if self.compute_segments:
            # Remove "End" markers for boundary detection
            segment_activation = torch.zeros(num_time_frames, dtype=torch.float32)
            segment_activation[segment_ticks] = 1.0
            segment_activation = widen_temporal_events(segment_activation, num_neighbors=2)
            sample["segment_activation"] = segment_activation

            # Create segment function labels
            segment_functions = []
            for label in segment_labels:
                if label != "End":
                    function = label.split(";")[0].strip()
                    segment_functions.append(function)
            
            # Create segment label activations (for each frame, which segment it belongs to)
            segment_label_activations = self._create_segment_label_activations(
                segment_ticks, segment_labels, num_time_frames
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
                    beat_ticks = [min(tick, num_time_frames - 1) for tick in beat_ticks]
                
                beat_activation = torch.zeros(num_time_frames, dtype=torch.float32)
                beat_activation[beat_ticks] = 1.0
                beat_activation = widen_temporal_events(beat_activation, num_neighbors=1)
                sample["beat_activation"] = torch.tensor(beat_activation, dtype=torch.float32)
            
            if self.compute_downbeats:
                downbeat_activation = torch.zeros(num_time_frames, dtype=torch.float32)
                downbeat_activation[measure_ticks] = 1.0
                downbeat_activation = widen_temporal_events(downbeat_activation, num_neighbors=1)
                sample["downbeat_activation"] = downbeat_activation

        return sample
    
    def _create_segment_label_activations(
        self, 
        segment_ticks: List[int], 
        segment_labels: List[str], 
        num_time_frames: int
    ) -> torch.Tensor:
        """Create frame-wise segment function labels."""
        segment_label_activations = torch.zeros(num_time_frames, dtype=torch.long)
        
        # Process segments
        for i in range(len(segment_labels) - 1):
            if segment_labels[i] == "End":
                continue
                
            start_tick = segment_ticks[i]
            end_tick = segment_ticks[i + 1]
            
            # Extract function and find its index in vocabulary
            function = segment_labels[i].split(";")[0].strip()
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
        if self.compute_segments:
            sample["segment_activation"] = torch.zeros(1)
            sample["segment_label_activations"] = torch.zeros(1, dtype=torch.long)
        if self.compute_beats:
            sample["beat_activation"] = torch.zeros(1)
        if self.compute_downbeats:
            sample["downbeat_activation"] = torch.zeros(1)
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