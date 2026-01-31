from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Union
import glob
from tqdm import tqdm
import json

import torch
from torch.utils.data import Dataset

from . import utils

class BaseMidiDataset(Dataset, ABC):
    """
    Abstract base class for MIDI datasets. Provides common functionality for
    parsing MIDI files, creating piano rolls, and computing SSLM features.
    """

    def __init__(
        self,
        target_ticks_per_beat: int = 4,
        instrument_overtones: bool = True,
        separate_drums: bool = True,
        transpose_augmentation: bool = False,
        use_sslms: bool = False,
        compute_segment_labels: bool = False,
        segment_function_vocab: Optional[List[str]] = None,
        midi_dir: Union[str, Path] = '',
        annotation_dir: Union[str, Path] = '',
        midi_files: Optional[List[str]] = None,
        piano_roll_dir: Optional[Union[str, Path]] = None,
        sslm_dir: Optional[Union[str, Path]] = None,

    ):
        self.target_ticks_per_beat = target_ticks_per_beat
        self.instrument_overtones = instrument_overtones
        self.separate_drums = separate_drums
        self.transpose_augmentation = transpose_augmentation
        self.use_sslms = use_sslms
        self.compute_segment_labels = compute_segment_labels
        self.segment_function_vocab = segment_function_vocab

        self.midi_dir = Path(midi_dir)
        self.annotation_dir = Path(annotation_dir)
        self.piano_roll_dir = Path(piano_roll_dir) if piano_roll_dir else None
        self.sslm_dir = Path(sslm_dir) if sslm_dir else None

        # Create cache directory if it doesn't exist
        if self.piano_roll_dir:
            self.piano_roll_dir.mkdir(parents=True, exist_ok=True)
        if self.sslm_dir:
            self.sslm_dir.mkdir(parents=True, exist_ok=True)

        # Get all annotation files and extract MIDI file IDs
        if midi_files is None:
            annotation_files = glob.glob(str(self.annotation_dir / "*_labels_coarse_qn.json"))
            self.midi_file_ids = [Path(f).stem.replace("_labels_coarse_qn", "") for f in annotation_files]
        else:
            self.midi_file_ids = midi_files

        # Filter out files that don't have both MIDI (or cached piano roll + sslms) and annotation
        valid_file_ids = []
        for file_id in tqdm(self.midi_file_ids, desc="Validating files"):
            midi_path = self.midi_dir / f"{file_id[0]}" / f"{file_id}.mid"
            annotation_path = self.annotation_dir / f"{file_id}_labels_coarse_qn.json"
            piano_roll_cache_path = utils.get_piano_roll_cache_path(file_id=file_id,
                                                                    piano_roll_dir=self.piano_roll_dir,
                                                                    target_ticks_per_beat=self.target_ticks_per_beat)
            sslm_cache_path = utils.get_sslm_cache_path(file_id=file_id, sslm_dir=self.sslm_dir,
                                                        target_ticks_per_beat=self.target_ticks_per_beat)
            is_valid = False
            if annotation_path.exists():
                if midi_path.exists():
                    is_valid = True
                if self.use_sslms:
                    if piano_roll_cache_path.exists() and sslm_cache_path.exists():
                        is_valid = True
                else:
                    if piano_roll_cache_path.exists():
                        is_valid = True
            if is_valid:
                valid_file_ids.append(file_id)

        self.midi_file_ids = valid_file_ids
        print(f"Found {len(self.midi_file_ids)} MIDI files with annotations")
        assert len(self.midi_file_ids) > 0, 'Found no valid MIDI! Aborting.'

        # Build segment function vocabulary if not provided
        if segment_function_vocab is None and compute_segment_labels:
            self.segment_function_vocab = self._build_segment_vocab()
        else:
            self.segment_function_vocab = segment_function_vocab

        # Precompute piano roll cache if caching is enabled
        if self.piano_roll_dir:
            self._precompute_piano_roll_cache()

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

    def _precompute_piano_roll_cache(self):
        """Precompute and cache all piano rolls."""
        if not self.piano_roll_dir:
            return

        # Check which files need to be cached
        files_to_cache = []
        for file_id in self.midi_file_ids:
            cache_path = utils.get_piano_roll_cache_path(file_id, self.piano_roll_dir, self.target_ticks_per_beat)
            if cache_path and not cache_path.exists():
                files_to_cache.append(file_id)

        if not files_to_cache:
            print(f"All {len(self.midi_file_ids)} piano rolls already cached")
            return

        print(f"Caching {len(files_to_cache)} piano rolls...")
        for file_id in tqdm(files_to_cache, desc="Caching piano rolls"):
            cache_path = utils.get_piano_roll_cache_path(file_id, self.piano_roll_dir, self.target_ticks_per_beat)
            if not cache_path or cache_path.exists():
                continue

            # Compute piano roll
            result = self._compute_piano_roll(file_id)
            if result is None:
                continue

            # Save to cache
            torch.save(result, cache_path)

    def _compute_piano_roll(self, file_id: str) -> Optional[dict[str, torch.Tensor]]:
        """Compute piano roll using base class method."""
        midi_path = self.midi_dir / f"{file_id[0]}" / f"{file_id}.mid"
        try:
            piano_roll = utils.create_piano_roll_fast(
                path_to_midi_file=midi_path,
                chroma=False,
                target_ticks_per_beat=self.target_ticks_per_beat,
            )
        except Exception as e:
            print(f"Error computing piano roll for {midi_path}: {e}")
            return None

        piano_roll['piano_roll'] = torch.from_numpy(piano_roll['piano_roll'])
        return piano_roll

    # def parse_and_create_piano_roll(
    #     self, midi_path: Union[str, Path]
    # ) -> Optional[Dict]:
    #     """
    #     Parse MIDI file and create piano roll.
    #
    #     Returns:
    #         Dictionary with 'piano_roll' (Tensor) and 'time_signatures' (list), or None if error
    #     """
    #     try:
    #         track_data, ticks_per_beat, time_signatures = parse_midi(midi_path)
    #
    #         piano_rolls = []
    #         for note_data in track_data.values():
    #             piano_roll = create_piano_roll(
    #                 note_data,
    #                 ticks_per_beat,
    #                 chroma=False,
    #                 target_ticks_per_beat=self.target_ticks_per_beat,
    #                 instrument_overtones=self.instrument_overtones,
    #                 separate_drums=self.separate_drums,
    #             )
    #             if piano_roll is not None:
    #                 piano_rolls.append(piano_roll)
    #
    #         if len(piano_rolls) == 0:
    #             return None
    #
    #         # Merge tracks
    #         actual_length = max(pr.shape[-1] for pr in piano_rolls)
    #         for i, pr in enumerate(piano_rolls):
    #             piano_rolls[i] = torch.nn.functional.pad(
    #                 torch.tensor(pr), (0, actual_length - pr.shape[-1])
    #             )
    #
    #         piano_roll = torch.stack(piano_rolls).sum(dim=0).clamp(0, 127)
    #         piano_roll = piano_roll.float() / 127.0
    #
    #         # Adjust time signatures for target ticks per beat
    #         adjusted_time_signatures = [
    #             (
    #                 int(round(tick_pos * self.target_ticks_per_beat / ticks_per_beat)),
    #                 numerator,
    #                 denominator,
    #             )
    #             for tick_pos, numerator, denominator in time_signatures
    #         ]
    #
    #         return {
    #             "piano_roll": piano_roll,
    #             "time_signatures": adjusted_time_signatures,
    #         }
    #
    #     except Exception as e:
    #         print(f"Error processing {midi_path}: {e}")
    #         return None
    #
    # def compute_sslm_features(
    #     self, piano_roll: torch.Tensor, L: int = 720
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Compute SSLM features from piano roll."""
    #     sslm_piano_roll = piano_roll.sum(dim=0, keepdim=True)
    #     return compute_sslms(sslm_piano_roll, L=L)
    #
    # def load_and_preprocess_annotations(
    #     self, annotation_path: Union[str, Path]
    # ) -> List[List]:
    #     """Load and preprocess segment annotations."""
    #     import json
    #
    #     with open(annotation_path, "r") as f:
    #         annotations = json.load(f)
    #     return preprocess_labels(annotations)

    def apply_transpose_augmentation(
        self, piano_roll: torch.Tensor, transpose_range: int = 6
    ) -> torch.Tensor:
        """Apply random transpose augmentation to piano roll."""
        transpose_amount = torch.randint(-transpose_range, transpose_range + 1, (1,)).item()
        transposed_piano_roll = piano_roll.clone()
        # Transpose first 2 channels only (non-drums)
        transposed_piano_roll[0] = torch.roll(transposed_piano_roll[0], int(transpose_amount), dims=-1)
        transposed_piano_roll[1] = torch.roll(transposed_piano_roll[1], int(transpose_amount), dims=-1)
        
        return transposed_piano_roll

    def get_piano_roll_dict(self, file_id: str):
        cache_path = utils.get_piano_roll_cache_path(file_id=file_id, piano_roll_dir=self.piano_roll_dir,
                                                     target_ticks_per_beat=self.target_ticks_per_beat)
        if cache_path and cache_path.exists():
            piano_roll_dict = torch.load(cache_path)
        else:
            piano_roll_dict = self._compute_piano_roll(file_id)
            if piano_roll_dict is None:
                raise ValueError(f'empty piano roll dict for {file_id}')
        return piano_roll_dict

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return dataset item."""
        pass
