from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Union

import torch
from torch.utils.data import Dataset

from .utils import parse_midi, create_piano_roll, compute_sslms
from .label_preprocessor import preprocess_labels


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
        compute_sslms: bool = False,
        segment_function_vocab: Optional[List[str]] = None,
    ):
        self.target_ticks_per_beat = target_ticks_per_beat
        self.instrument_overtones = instrument_overtones
        self.separate_drums = separate_drums
        self.transpose_augmentation = transpose_augmentation
        self.compute_sslms = compute_sslms
        self.segment_function_vocab = segment_function_vocab

    def parse_and_create_piano_roll(
        self, midi_path: Union[str, Path]
    ) -> Optional[Dict]:
        """
        Parse MIDI file and create piano roll.

        Returns:
            Dictionary with 'piano_roll' (Tensor) and 'time_signatures' (list), or None if error
        """
        try:
            track_data, ticks_per_beat, time_signatures = parse_midi(midi_path)

            piano_rolls = []
            for note_data in track_data.values():
                piano_roll = create_piano_roll(
                    note_data,
                    ticks_per_beat,
                    chroma=False,
                    target_ticks_per_beat=self.target_ticks_per_beat,
                    instrument_overtones=self.instrument_overtones,
                    separate_drums=self.separate_drums,
                )
                if piano_roll is not None:
                    piano_rolls.append(piano_roll)

            if len(piano_rolls) == 0:
                return None

            # Merge tracks
            actual_length = max(pr.shape[-1] for pr in piano_rolls)
            for i, pr in enumerate(piano_rolls):
                piano_rolls[i] = torch.nn.functional.pad(
                    torch.tensor(pr), (0, actual_length - pr.shape[-1])
                )

            piano_roll = torch.stack(piano_rolls).sum(dim=0).clamp(0, 127)
            piano_roll = piano_roll.float() / 127.0

            # Adjust time signatures for target ticks per beat
            adjusted_time_signatures = [
                (
                    int(round(tick_pos * self.target_ticks_per_beat / ticks_per_beat)),
                    numerator,
                    denominator,
                )
                for tick_pos, numerator, denominator in time_signatures
            ]

            return {
                "piano_roll": piano_roll,
                "time_signatures": adjusted_time_signatures,
            }

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None

    def compute_sslm_features(
        self, piano_roll: torch.Tensor, L: int = 720
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute SSLM features from piano roll."""
        sslm_piano_roll = piano_roll.sum(dim=0, keepdim=True)
        return compute_sslms(sslm_piano_roll, L=L)

    def load_and_preprocess_annotations(
        self, annotation_path: Union[str, Path]
    ) -> List[List]:
        """Load and preprocess segment annotations."""
        import json

        with open(annotation_path, "r") as f:
            annotations = json.load(f)
        return preprocess_labels(annotations)

    def apply_transpose_augmentation(
        self, piano_roll: torch.Tensor, transpose_range: int = 6
    ) -> torch.Tensor:
        """Apply random transpose augmentation to piano roll."""
        transpose_amount = torch.randint(-transpose_range, transpose_range + 1, (1,)).item()
        return torch.roll(piano_roll, int(transpose_amount), dims=-2)

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return dataset item."""
        pass
