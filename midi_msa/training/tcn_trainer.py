import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..data.tcn_dataset import TCNMidiDataset
from ..data.label_preprocessor import LABEL_MAP


class TCNTrainer(BaseTrainer):
    """Trainer for TCN (sequence-based) method."""

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)

        # Build segment vocabulary
        self.label_map = sorted(list(set(LABEL_MAP.values())))

    def lower_is_better(self) -> bool:
        return True  # Loss should be minimized

    def get_dataloaders(self) -> Tuple:
        """Create dataloaders for TCN training."""
        # Load split files if provided
        if self.cfg.split_files:
            with open(self.cfg.split_files[0], "r") as f:
                splits = json.load(f)
            train_files = splits.get("train", [])
            val_files = splits.get("val", [])
        else:
            # Auto-split from all files
            import glob
            from pathlib import Path

            all_files = glob.glob(
                str(self.cfg.annotation_dir / "*_labels_coarse_qn.json")
            )
            file_ids = [
                Path(f).stem.replace("_labels_coarse_qn", "") for f in all_files
            ]
            split_idx = int(len(file_ids) * (1 - self.cfg.val_split))
            train_files = file_ids[:split_idx]
            val_files = file_ids[split_idx:]

        # Create datasets
        dataset_args = {
            "midi_dir": self.cfg.midi_dir,
            "annotation_dir": self.cfg.annotation_dir,
            "piano_roll_dir": self.cfg.piano_roll_dir,
            "segment_function_vocab": self.label_map,
            "target_ticks_per_beat": self.cfg.target_ticks_per_beat,
            "compute_beats": self.cfg.compute_beats,
            "compute_downbeats": self.cfg.compute_downbeats,
            "compute_segments": self.cfg.compute_segments,
            "instrument_overtones": self.cfg.instrument_overtones,
            "separate_drums": self.cfg.separate_drums,
            "transpose_augmentation": self.cfg.transpose_augmentation,
        }

        train_dataset = TCNMidiDataset(
            midi_files=train_files, sslms_dir=self.cfg.sslm_dir, **dataset_args
        )
        val_dataset = TCNMidiDataset(
            midi_files=val_files, sslms_dir=self.cfg.sslm_dir, **dataset_args
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        return train_loader, (val_loader,)

    def compute_loss(self, model_output, targets: Dict[str, torch.Tensor]) -> Dict:
        """Compute multi-task losses."""
        losses = {}
        weighted_losses = {}

        if "beat_activation" in targets:
            beat_loss = nn.functional.binary_cross_entropy_with_logits(
                model_output.beat_output, targets["beat_activation"]
            )
            losses["beat_loss"] = beat_loss
            weighted_losses["beat_loss"] = beat_loss * self.cfg.beat_loss_weight

        if "downbeat_activation" in targets:
            downbeat_loss = nn.functional.binary_cross_entropy_with_logits(
                model_output.downbeat_output, targets["downbeat_activation"]
            )
            losses["downbeat_loss"] = downbeat_loss
            weighted_losses["downbeat_loss"] = (
                downbeat_loss * self.cfg.downbeat_loss_weight
            )

        if "segment_activation" in targets:
            segment_loss = nn.functional.binary_cross_entropy_with_logits(
                model_output.segment_output, targets["segment_activation"]
            )
            losses["segment_loss"] = segment_loss
            weighted_losses["segment_loss"] = segment_loss * self.cfg.section_loss_weight

        if "segment_label_activations" in targets:
            _, num_classes, _ = model_output.function_outputs.shape
            function_outputs = (
                model_output.function_outputs.permute(0, 2, 1).reshape(-1, num_classes)
            )
            function_targets = targets["segment_label_activations"].reshape(-1)

            function_loss = nn.functional.cross_entropy(
                function_outputs, function_targets, ignore_index=-100
            )
            losses["function_loss"] = function_loss
            weighted_losses["function_loss"] = (
                function_loss * self.cfg.function_loss_weight
            )

        total_loss = sum(weighted_losses.values())
        losses["total_loss"] = total_loss

        return losses

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            piano_rolls = batch["piano_roll"].to(self.device)
            sslm_near = batch.get("sslm_near")
            sslm_far = batch.get("sslm_far")

            if sslm_near is not None:
                sslm_near = sslm_near.to(self.device)
            if sslm_far is not None:
                sslm_far = sslm_far.to(self.device)

            targets = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k not in ["piano_roll", "sslm_near", "sslm_far", "measure_ticks"]
            }

            self.optimizer.zero_grad()
            outputs = self.model(piano_rolls, sslm_near=sslm_near, sslm_far=sslm_far)

            losses = self.compute_loss(outputs, targets)

            losses["total_loss"].backward()

            # Gradient clipping
            if self.cfg.clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_norm)

            self.optimizer.step()

            total_loss += losses["total_loss"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def validate_epoch(self, val_loaders) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_loader = val_loaders[0]

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                piano_rolls = batch["piano_roll"].to(self.device)
                sslm_near = batch.get("sslm_near")
                sslm_far = batch.get("sslm_far")

                if sslm_near is not None:
                    sslm_near = sslm_near.to(self.device)
                if sslm_far is not None:
                    sslm_far = sslm_far.to(self.device)

                targets = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k not in ["piano_roll", "sslm_near", "sslm_far", "measure_ticks"]
                }

                outputs = self.model(
                    piano_rolls, sslm_near=sslm_near, sslm_far=sslm_far
                )
                losses = self.compute_loss(outputs, targets)

                total_loss += losses["total_loss"].item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def get_val_metric_for_early_stopping(self, val_metrics: Dict[str, float]) -> float:
        """Use validation loss for early stopping."""
        return val_metrics["loss"]
