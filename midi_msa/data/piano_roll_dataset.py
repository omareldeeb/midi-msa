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
    ):
        self.piano_roll_patches = piano_roll_patches
        self.metadata_df = metadata_df
        self.normalize = normalize
        self.transpose_augmentation = transpose_augmentation
        self.target_ticks_per_beat = target_ticks_per_beat
        self.num_targets = num_targets

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        sample = self.metadata_df.loc[idx]
        piano_roll = self.piano_roll_patches[sample["piano_roll_idx"]]
        patch = piano_roll[..., sample["from"].int():sample["to"].int()]

        center = sample["from"] + (sample["to"] - sample["from"]) / 2
        nearest_segment_boundary = sample["nearest_segment_boundary"]

        # targets: boundary at center? boundary within (2, 4, 8) bars of center?
        # TODO: assumes a single segment boundary per patch
        main_target = [sample["is_segment_boundary"]]
        additional_targets = [(nearest_segment_boundary - center).abs() <= 2**i * self.target_ticks_per_beat * 4 for i in range(self.num_targets - 1)]
        
        targets = torch.tensor(main_target + additional_targets).to(torch.float32)

        if self.normalize:
            patch = patch / patch.max() 

        if self.transpose_augmentation:
            patch = transpose_augmentation(patch)

        return patch, targets
    
    def metadata_at(self, idx):
        sample = self.metadata_df.loc[idx]
        return {
            "filename": sample["filename"],
            "from": sample["from"],
            "to": sample["to"]
        }