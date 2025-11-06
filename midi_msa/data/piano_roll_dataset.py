from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.utils.data import Dataset


def transpose_augmentation(piano_roll, transpose_range=6):
    transpose_amount = torch.randint(-transpose_range, transpose_range, ())
    return torch.roll(piano_roll, transpose_amount.item(), dims=-2) # type: ignore


class PianoRollDataset(Dataset):
    def __init__(
        self,
        piano_roll_patches,
        metadata_df,
        normalize=False,
        transpose_augmentation=True,
        target_ticks_per_beat=4,
        num_targets=1,  # Additional targets within 2**i bars of center where i < NUM_TARGETS
        sslm_near_patches=None,
        sslm_far_patches=None,
        segment_function_vocab: Optional[List[str]] = None,
    ):
        self.piano_roll_patches = piano_roll_patches
        self.metadata_df = metadata_df
        self.normalize = normalize
        self.transpose_augmentation = transpose_augmentation
        self.target_ticks_per_beat = target_ticks_per_beat
        self.num_targets = num_targets
        self.sslm_near_patches = sslm_near_patches
        self.sslm_far_patches = sslm_far_patches
        self.segment_function_vocab = segment_function_vocab

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        sample = self.metadata_df.loc[idx]
        piano_roll = self.piano_roll_patches[sample["piano_roll_idx"]].squeeze()
        from_tick, to_tick = sample["from"].int(), sample["to"].int()
        piano_roll_patch = piano_roll[..., from_tick:to_tick]

        center = from_tick + (to_tick - from_tick) / 2
        nearest_segment_boundary = sample["nearest_segment_boundary"]

        # targets: boundary at center? boundary within (2, 4, 8) bars of center?
        # TODO: assumes a single segment boundary per patch
        main_target = [sample["is_segment_boundary"]]
        additional_targets = [(nearest_segment_boundary - center).abs() <= 2**i * self.target_ticks_per_beat * 4 for i in range(self.num_targets - 1)]

        targets = torch.tensor(main_target + additional_targets).to(torch.float32)

        if self.normalize:
            piano_roll_patch = piano_roll_patch / piano_roll_patch.max()

        if self.transpose_augmentation:
            piano_roll_patch = transpose_augmentation(piano_roll_patch)

        item = {
            "piano_roll_patch": piano_roll_patch,
            "targets": targets,
        }

        # Add segment label target if available and enabled
        if self.segment_function_vocab is not None and "segment_label" in sample:
            segment_label = sample["segment_label"]
            if segment_label in self.segment_function_vocab:
                segment_label_idx = self.segment_function_vocab.index(segment_label)
            else:
                # Default to instrumental if unknown (or first label if Instrumental not in vocab)
                if "Instrumental" in self.segment_function_vocab:
                    segment_label_idx = self.segment_function_vocab.index("Instrumental")
                else:
                    segment_label_idx = 0  # Fallback to first label
            item["segment_label_target"] = torch.tensor(segment_label_idx, dtype=torch.long)

        if self.sslm_near_patches is not None:
            sslm_near = self.sslm_near_patches[sample["sslm_near_patch_idx"]]
            if len(sslm_near.shape) == 2:
                sslm_near = sslm_near.unsqueeze(0)  # Add channel dimension if missing
            sslm_near_patch = sslm_near[..., from_tick:to_tick]
            item["sslm_near_patch"] = sslm_near_patch

        if self.sslm_far_patches is not None:
            sslm_far = self.sslm_far_patches[sample["sslm_far_patch_idx"]]
            if len(sslm_far.shape) == 2:
                sslm_far = sslm_far.unsqueeze(0)  # Add channel dimension if missing
            sslm_far_patch = sslm_far[..., from_tick:to_tick]
            item["sslm_far_patch"] = sslm_far_patch

        return item

    def metadata_at(self, idx):
        sample = self.metadata_df.loc[idx]
        return {
            "filename": sample["filename"],
            "from": sample["from"],
            "to": sample["to"]
        }