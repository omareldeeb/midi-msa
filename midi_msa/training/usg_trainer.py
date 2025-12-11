from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..data.piano_roll_dataset import PianoRollDataset
from ..data.utils import get_piano_roll_patches
from ..evaluation.metrics import compute_metrics
from ..models.mobilenet_boundary_classifier import SEGMENT_LABEL_VOCAB


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
        segment_vocab = SEGMENT_LABEL_VOCAB if self.cfg.predict_segment_label else None

        # Load patches
        patch_data = get_piano_roll_patches(
            data_dir=self.cfg.data_dir,
            window_half_ticks=self.cfg.window_half_ticks,
            positive_oversampling_factor=self.cfg.positive_oversampling_factor,
            negative_undersampling_factor=self.cfg.negative_undersampling_factor,
            pad_boundary_patches=self.cfg.pad_boundary_patches,
            return_sslm_near=self.cfg.use_sslm_near,
            return_sslm_far=self.cfg.use_sslm_far,
        )

        import pandas as pd

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

        return train_loader, (val_loader_tubb, val_loader_non_tubb)

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
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

        avg_loss = total_loss / len(train_loader)
        return {"loss": avg_loss}

    def validate_epoch(self, val_loaders) -> Dict[str, float]:
        """Validate on both tubb and non-tubb sets."""
        self.model.eval()

        val_loader_tubb, val_loader_non_tubb = val_loaders

        # Validate on tubb
        val_outputs_tubb, val_targets_tubb = [], []
        val_loss_tubb = 0.0

        with torch.no_grad():
            for batch in val_loader_tubb:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
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

                val_outputs_tubb.append(output["boundary_logits"])
                val_targets_tubb.append(batch["targets"])
                val_loss_tubb += loss.item()

        # Validate on non-tubb
        val_outputs_non_tubb, val_targets_non_tubb = [], []
        val_loss_non_tubb = 0.0

        with torch.no_grad():
            for batch in val_loader_non_tubb:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
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

                val_outputs_non_tubb.append(output["boundary_logits"])
                val_targets_non_tubb.append(batch["targets"])
                val_loss_non_tubb += loss.item()

        # Compute metrics
        metrics_tubb = compute_metrics(
            torch.cat(val_outputs_tubb), torch.cat(val_targets_tubb)
        )
        metrics_non_tubb = compute_metrics(
            torch.cat(val_outputs_non_tubb), torch.cat(val_targets_non_tubb)
        )

        val_loss_tubb /= len(val_loader_tubb)
        val_loss_non_tubb /= len(val_loader_non_tubb)

        # Compute F1
        f1_tubb = (
            2
            * metrics_tubb["precision_0"]
            * metrics_tubb["recall_0"]
            / (metrics_tubb["precision_0"] + metrics_tubb["recall_0"] + 1e-8)
        )
        f1_non_tubb = (
            2
            * metrics_non_tubb["precision_0"]
            * metrics_non_tubb["recall_0"]
            / (metrics_non_tubb["precision_0"] + metrics_non_tubb["recall_0"] + 1e-8)
        )

        return {
            "loss": (val_loss_tubb + val_loss_non_tubb) / 2,
            "loss_tubb": val_loss_tubb,
            "loss_non_tubb": val_loss_non_tubb,
            "f1_tubb": f1_tubb,
            "f1_non_tubb": f1_non_tubb,
            "f1_avg": (f1_tubb + f1_non_tubb) / 2,
            "precision_tubb": metrics_tubb["precision_0"],
            "precision_non_tubb": metrics_non_tubb["precision_0"],
            "recall_tubb": metrics_tubb["recall_0"],
            "recall_non_tubb": metrics_non_tubb["recall_0"],
        }

    def get_val_metric_for_early_stopping(self, val_metrics: Dict[str, float]) -> float:
        """Use average loss for early stopping."""
        return val_metrics["loss"]
