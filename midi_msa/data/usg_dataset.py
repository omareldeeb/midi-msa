from .base_dataset import BaseMidiDataset
from pathlib import Path
from typing import List, Dict, Optional, Union
from ..data import utils
import torch
from tqdm import tqdm
from .label_preprocessor import preprocess_labels
import bisect
import torch.nn.functional as F


class USGMidiDataset(BaseMidiDataset):
    def __init__(self,
                 midi_dir: Union[str, Path],
                 annotation_dir: Union[str, Path],
                 extra_midi_dir: Union[str, Path],
                 midi_files: Optional[List[str]] = None,
                 target_ticks_per_beat: int = 4,
                 window_half_ticks: int = 256,
                 pad_boundary_patches: bool = True,
                 segment_function_vocab: Optional[List[str]] = None,
                 label_map: Optional[Dict[str, str]] = None,
                 compute_segment_labels: bool = True,
                 instrument_overtones: bool = True,
                 separate_drums: bool = True,
                 use_sslms: bool = True,
                 piano_roll_dir: Optional[Union[str, Path]] = None,
                 sslm_dir: Optional[Union[str, Path]] = None,
                 transpose_augmentation: bool = True,
                 positive_oversampling_factor: int = 2,
                 negative_undersampling_factor: int = 1,
                 normalize_patches: bool = False,
                 num_targets: int = 1,
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
            extra_midi_dir=extra_midi_dir,
            midi_files=midi_files,
            piano_roll_dir=piano_roll_dir,
            sslm_dir=sslm_dir,
        )

        self.window_half_ticks = window_half_ticks
        self.pad_boundary_patches = pad_boundary_patches
        self.positive_oversampling_factor = positive_oversampling_factor
        self.negative_undersampling_factor = negative_undersampling_factor
        self.normalize_patches = normalize_patches
        self.num_targets = num_targets

        # precompute sslms
        if self.use_sslms:
            for file_id in tqdm(self.midi_file_ids, desc="Checking for/computing SSLMs"):
                sslm_cache_path = utils.get_sslm_cache_path(file_id, self.sslm_dir, self.target_ticks_per_beat)
                if (not sslm_cache_path) or (not sslm_cache_path.exists()):
                    # Merge piano roll across channels for SSLM computation by summing
                    midi_path = utils.get_midi_path(file_id=file_id, midi_dirs=[self.midi_dir, self.extra_midi_dir])
                    sslm_near, sslm_far = utils.compute_sslms_from_midi_path(p=midi_path,
                                                                             target_ticks_per_beat=self.target_ticks_per_beat)
                    if sslm_cache_path:
                        torch.save({"sslm_near": sslm_near, "sslm_far": sslm_far}, sslm_cache_path)

        # precompute patch data
        self.patch_metadata_list = []
        for file_id in self.midi_file_ids:
            piano_roll_dict = self.get_piano_roll_dict(file_id=file_id)
            measure_ticks = piano_roll_dict['measure_ticks']
            piano_roll = piano_roll_dict['piano_roll']
            annotation_path = self.annotation_dir / f"{file_id}_functions_qn.json"
            annotations = utils.load_annotation(annotation_path)
            annotations = preprocess_labels(annotations, label_map=self.label_map)
            segment_qns = [ann[0] for ann in annotations]
            segment_labels = [ann[1] for ann in annotations]
            n_ticks = piano_roll.shape[-1]
            segment_ticks = [round(x * self.target_ticks_per_beat) for x in segment_qns]
            segment_ticks = [x for x in segment_ticks if x < n_ticks]
            for m in measure_ticks:
                if m < n_ticks:
                    patch_metadata = {'file_id': file_id,
                                      'start_tick': m - self.window_half_ticks,
                                      'end_tick': m + self.window_half_ticks,
                                      'center_tick': m,
                                      'is_boundary': m in segment_ticks,
                                      'label': None,
                                      'n_ticks_in_file': n_ticks,
                                      }
                    if self.compute_segment_labels:
                        label_idx = bisect.bisect_right(segment_ticks, m) - 1
                        if label_idx == -1:
                            patch_metadata['label'] = 'Start'
                        else:
                            patch_metadata['label'] = segment_labels[label_idx]
                    if self.pad_boundary_patches or (patch_metadata['start_tick'] >= 0 and patch_metadata['end_tick'] < n_ticks):
                        n_times = self.positive_oversampling_factor if patch_metadata['is_boundary'] else self.negative_undersampling_factor
                        for _ in range(n_times):
                            self.patch_metadata_list.append(patch_metadata)

    def __len__(self):
        return len(self.patch_metadata_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch_metadata = self.patch_metadata_list[idx]
        piano_roll_dict = self.get_piano_roll_dict(file_id=patch_metadata['file_id'])
        piano_roll = piano_roll_dict['piano_roll']

        if self.transpose_augmentation:
            piano_roll = self.apply_transpose_augmentation(piano_roll)

        padded_piano_roll = F.pad(piano_roll, (self.window_half_ticks, self.window_half_ticks),
                                  mode='constant', value=0)
        piano_roll_patch = padded_piano_roll[..., patch_metadata['center_tick']: patch_metadata['center_tick'] + 2*self.window_half_ticks]

        # compute_piano_roll now always returns 3 channels: non-drums, overtones, drums
        # consolidate output format here
        if not self.separate_drums and self.instrument_overtones:
            piano_roll_patch = torch.stack([
                piano_roll_patch[0] + piano_roll_patch[2],
                piano_roll_patch[1],
                torch.zeros_like(piano_roll_patch[0])
            ])
        elif self.separate_drums and not self.instrument_overtones:
            piano_roll_patch = torch.stack([
                piano_roll_patch[0],
                torch.zeros_like(piano_roll_patch[0]),
                piano_roll_patch[2]
            ])
        elif not self.separate_drums and not self.instrument_overtones:
            piano_roll_patch = torch.stack([
                piano_roll_patch[0] + piano_roll_patch[2],
                torch.zeros_like(piano_roll_patch[0]),
                torch.zeros_like(piano_roll_patch[0])
            ])
        piano_roll_patch = torch.clip(piano_roll_patch, 0.0, 1.0)

        if self.normalize_patches:
            piano_roll_patch = piano_roll_patch / piano_roll_patch.max()

        targets = [int(patch_metadata['is_boundary'])]
        if self.num_targets != 1:
            raise NotImplementedError('USGMidiDataset only supports num_targets = 1')
        targets = torch.tensor(targets).to(torch.float32)

        item = {
            'piano_roll_patch': piano_roll_patch,
            'targets': targets
        }

        if self.segment_function_vocab is not None and self.compute_segment_labels and patch_metadata['label'] is not None:
            label = patch_metadata['label']
            label_idx = self.segment_function_vocab.index(label)
            item['segment_label_target'] = torch.tensor(label_idx, dtype=torch.long)

        if self.use_sslms:
            sslm_cache_path = utils.get_sslm_cache_path(file_id=patch_metadata['file_id'],
                                                        sslm_dir=self.sslm_dir,
                                                        target_ticks_per_beat=self.target_ticks_per_beat)
            sslm_data = torch.load(sslm_cache_path)

            sslm_near = sslm_data["sslm_near"]
            if len(sslm_near.shape) == 2:
                sslm_near = sslm_near.unsqueeze(0)  # Add channel dimension if missing
            padded_sslm_near = F.pad(sslm_near, (self.window_half_ticks, self.window_half_ticks),
                                     mode='constant', value=0)
            sslm_near_patch = padded_sslm_near[..., patch_metadata['center_tick']: patch_metadata['center_tick'] + 2*self.window_half_ticks]
            item["sslm_near_patch"] = sslm_near_patch

            sslm_far = sslm_data["sslm_far"]
            if len(sslm_far.shape) == 2:
                sslm_far = sslm_far.unsqueeze(0)  # Add channel dimension if missing
            padded_sslm_far = F.pad(sslm_far, (self.window_half_ticks, self.window_half_ticks),
                                    mode='constant', value=0)
            sslm_far_patch = padded_sslm_far[..., patch_metadata['center_tick']: patch_metadata['center_tick'] + 2*self.window_half_ticks]
            item["sslm_far_patch"] = sslm_far_patch

        item['patch_metadata'] = patch_metadata
        return item
