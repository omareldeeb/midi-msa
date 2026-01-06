import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..data.piano_roll_dataset import PianoRollDataset
from ..data.utils import get_piano_roll_patches, create_piano_roll_patch_data
from ..evaluation.metrics import compute_metrics
from ..data.label_preprocessor import LABEL_MAP


class USGTrainer(BaseTrainer):
    """Trainer for USG (patch-based) method."""

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)

        self.boundary_criterion = nn.BCEWithLogitsLoss()
        self.segment_criterion = (
            nn.CrossEntropyLoss() if cfg.predict_segment_label else None
        )

    def lower_is_better(self) -> bool:
        return True  # Loss should be minimized

    def get_dataloaders(self) -> Tuple:
        """Create dataloaders for USG training."""
        from pathlib import Path
        import json

        files_dict = None
        if self.cfg.split_files and Path(self.cfg.split_files[0]).exists():
            with open(self.cfg.split_files[0], 'r') as f:
                files_dict = json.load(f)
        else:
            # Random split
            val_split = self.cfg.val_split if hasattr(self.cfg, 'val_split') else 0.1
            print(f"No split_files provided, using random split with val_split={val_split}.")

            all_midi_files = []
            for root, _, files in os.walk(self.cfg.midi_dir):
                for file in files:
                    if not file.startswith('.') and (file.endswith('.mid') or file.endswith('.midi')):
                        all_midi_files.append(os.path.join(root, file))
            np.random.shuffle(all_midi_files)
            num_val = int(len(all_midi_files) * val_split)
            files_dict = {
                "train": all_midi_files[num_val:],
                "val": all_midi_files[:num_val],
            }

        return self._create_dataloaders(files_dict)

    def get_dataloaders_for_fold(self, split_file: str) -> Tuple:
        """Create dataloaders for a specific fold."""
        import json

        with open(split_file, 'r') as f:
            files_dict = json.load(f)

        return self._create_dataloaders(files_dict)

    def _create_dataloaders(self, files_dict: dict) -> Tuple:
        """Create train and validation dataloaders from a files dictionary."""
        import pandas as pd

        patch_data = create_piano_roll_patch_data(
            midi_dir=self.cfg.midi_dir,
            files_dict=files_dict,
            markers_qn_path=self.cfg.markers_qn_path,
            measures_qn_path=self.cfg.measures_qn_path,
            annotation_dir=self.cfg.annotation_dir,
            piano_roll_dir=self.cfg.piano_roll_dir,
            sslm_dir=self.cfg.sslm_dir,
            window_half_ticks=self.cfg.window_half_ticks,
            target_ticks_per_beat=self.cfg.target_ticks_per_beat,
            instrument_overtones=self.cfg.instrument_overtones,
            separate_drums=self.cfg.separate_drums,
            positive_oversampling_factor=self.cfg.positive_oversampling_factor,
            negative_undersampling_factor=self.cfg.negative_undersampling_factor,
            return_sslm_near=self.cfg.use_sslm_near,
            return_sslm_far=self.cfg.use_sslm_far,
        )

        piano_rolls = patch_data.piano_rolls
        metadata_dict = patch_data.patch_metadata
        sslm_near_patches = patch_data.sslm_near_patches
        sslm_far_patches = patch_data.sslm_far_patches
        metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="index").sample(
            frac=1
        )

        # Split into train/val
        metadata_train = metadata_df[
            metadata_df["key"].isin(["tubb_train", "non_tubb_train"])
        ]
        metadata_val_tubb = metadata_df[metadata_df["key"] == "tubb_val"]
        metadata_val_non_tubb = metadata_df[metadata_df["key"] == "non_tubb_val"]

        metadata_train.reset_index(drop=True, inplace=True)
        metadata_val_tubb.reset_index(drop=True, inplace=True)
        metadata_val_non_tubb.reset_index(drop=True, inplace=True)

        # Create datasets
        segment_vocab = sorted(list(set(LABEL_MAP.values())))
        dataset_train = PianoRollDataset(
            piano_rolls,
            metadata_train,
            normalize=self.cfg.patch_normalize,
            num_targets=self.cfg.num_targets,
            sslm_near_patches=sslm_near_patches,
            sslm_far_patches=sslm_far_patches,
            segment_function_vocab=segment_vocab,
        )

        dataset_val_tubb = PianoRollDataset(
            piano_rolls,
            metadata_val_tubb,
            normalize=self.cfg.patch_normalize,
            num_targets=self.cfg.num_targets,
            sslm_near_patches=sslm_near_patches,
            sslm_far_patches=sslm_far_patches,
            segment_function_vocab=segment_vocab,
        )

        dataset_val_non_tubb = PianoRollDataset(
            piano_rolls,
            metadata_val_non_tubb,
            normalize=self.cfg.patch_normalize,
            num_targets=self.cfg.num_targets,
            sslm_near_patches=sslm_near_patches,
            sslm_far_patches=sslm_far_patches,
            segment_function_vocab=segment_vocab,
        )

        # Create dataloaders
        train_loader = DataLoader(
            dataset_train, batch_size=self.cfg.batch_size, shuffle=True
        )
        val_loader_tubb = DataLoader(
            dataset_val_tubb, batch_size=self.cfg.batch_size, shuffle=False
        )
        val_loader_non_tubb = DataLoader(
            dataset_val_non_tubb, batch_size=self.cfg.batch_size, shuffle=False
        )

        # Return val loaders as iterable of (name, dataloader) tuples
        val_loaders = [("tubb", val_loader_tubb), ("non_tubb", val_loader_non_tubb)]

        return train_loader, val_loaders

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in (enumerate(pbar)):
            batch = {
                k: v.to(torch.float32).to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()
            output = self.model(
                batch["piano_roll_patch"],
                batch.get("sslm_near_patch"),
                batch.get("sslm_far_patch"),
            )

            boundary_loss = self.boundary_criterion(
                output["boundary_logits"], batch["targets"].float()
            )
            loss = boundary_loss

            if "segment_label_logits" in output and "segment_label_target" in batch:
                segment_loss = self.segment_criterion(
                    output["segment_label_logits"], batch["segment_label_target"]
                )
                loss = loss + self.cfg.segment_label_loss_weight * segment_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # update progress bar with current loss and average loss
            pbar.set_postfix({
                "batch_loss": loss.item(),
                "avg_loss": total_loss / (batch_idx + 1)
            })

        avg_loss = total_loss / len(train_loader)
        return {"loss": avg_loss}

    def validate_epoch(self, val_loaders) -> Dict[str, float]:
        """Validate on all validation loaders.

        Args:
            val_loaders: Iterable of (name, dataloader) tuples or dict mapping names to dataloaders.
        """
        self.model.eval()

        # Support both dict and iterable of tuples
        if isinstance(val_loaders, dict):
            val_loaders = val_loaders.items()

        all_metrics = {}
        all_losses = []
        all_f1s = []

        for loader_name, val_loader in val_loaders:
            if len(val_loader) == 0:
                continue  # Skip empty loaders
            val_outputs, val_targets = [], []
            total_loss = 0.0

            pbar = tqdm(val_loader, desc=f"Validating ({loader_name})")
            with torch.no_grad():
                for batch_idx, batch in enumerate(pbar):
                    batch = {
                        k: v.to(torch.float32).to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    output = self.model(
                        batch["piano_roll_patch"],
                        batch.get("sslm_near_patch"),
                        batch.get("sslm_far_patch"),
                    )

                    boundary_loss = self.boundary_criterion(
                        output["boundary_logits"], batch["targets"].float()
                    )
                    loss = boundary_loss

                    if "segment_label_logits" in output and "segment_label_target" in batch:
                        segment_loss = self.segment_criterion(
                            output["segment_label_logits"], batch["segment_label_target"]
                        )
                        loss = loss + self.cfg.segment_label_loss_weight * segment_loss

                    val_outputs.append(output["boundary_logits"])
                    val_targets.append(batch["targets"])
                    total_loss += loss.item()

                    # Update progress bar with current and average loss
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({
                        "batch_loss": loss.item(),
                        "avg_loss": avg_loss
                    })

            # Compute metrics for this loader
            avg_loss = total_loss / len(val_loader)
            metrics = compute_metrics(
                torch.sigmoid(torch.cat(val_outputs)), torch.cat(val_targets)
            )

            # Compute F1
            f1 = (
                2
                * metrics["precision_0"]
                * metrics["recall_0"]
                / (metrics["precision_0"] + metrics["recall_0"] + 1e-8)
            )

            # Store metrics with loader name prefix
            all_metrics[f"loss_{loader_name}"] = avg_loss
            all_metrics[f"f1_{loader_name}"] = f1
            all_metrics[f"precision_{loader_name}"] = metrics["precision_0"]
            all_metrics[f"recall_{loader_name}"] = metrics["recall_0"]

            all_losses.append(avg_loss)
            all_f1s.append(f1)

        # Compute aggregate metrics
        all_metrics["loss"] = sum(all_losses) / len(all_losses)
        all_metrics["f1_avg"] = sum(all_f1s) / len(all_f1s)

        return all_metrics

    def get_val_metric_for_early_stopping(self, val_metrics: Dict[str, float]) -> float:
        """Use average loss for early stopping."""
        return val_metrics["loss"]
