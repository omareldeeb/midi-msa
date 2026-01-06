import json
from typing import Dict, Tuple

import mir_eval.segment
import mir_eval.util
import numpy as np
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
        assert self.cfg.annotation_dir is not None, "annotation_dir must be specified for TCN method."

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
                str(Path(self.cfg.annotation_dir) / "*_labels_coarse_qn.json")
            )
            np.random.shuffle(all_files)

            file_ids = [
                Path(f).stem.replace("_labels_coarse_qn", "") for f in all_files
            ]
            split_idx = int(len(file_ids) * (1 - self.cfg.val_split))
            train_files = file_ids[:split_idx]
            val_files = file_ids[split_idx:]

        return self._create_dataloaders(train_files, val_files)

    def get_dataloaders_for_fold(self, split_file: str) -> Tuple:
        """Create dataloaders for a specific fold."""
        assert self.cfg.annotation_dir is not None, "annotation_dir must be specified for TCN method."

        with open(split_file, "r") as f:
            splits = json.load(f)
        train_files = splits.get("train", [])
        val_files = splits.get("val", [])

        return self._create_dataloaders(train_files, val_files)

    def _create_dataloaders(self, train_files, val_files) -> Tuple:
        """Create train and validation dataloaders from file lists."""
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
            batch_size=1,  # TCN processes full sequences; batch size 1
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # TCN processes full sequences; batch size 1
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

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
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

            # update progress bar with current loss and average loss
            pbar.set_postfix({
                "batch_loss": losses["total_loss"].item(),
                "avg_loss": total_loss / num_batches,
            })

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def validate_epoch(self, val_loaders) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_loader = val_loaders[0]

        total_loss = 0.0
        num_batches = 0

        # Metrics accumulators
        total_boundary_prec = 0.0
        total_boundary_recall = 0.0
        total_boundary_f1 = 0.0
        total_pairwise_prec = 0.0
        total_pairwise_recall = 0.0
        total_pairwise_f1 = 0.0
        num_boundary_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                piano_rolls = batch["piano_roll"].to(self.device)
                sslm_near = batch.get("sslm_near")
                sslm_far = batch.get("sslm_far")
                measure_ticks = batch.get("measure_ticks")

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

                # Compute boundary and pairwise metrics
                if measure_ticks is not None and "segment_activation" in targets:
                    boundaries_pred = torch.sigmoid(outputs.segment_output).squeeze()
                    boundaries_target = targets["segment_activation"].squeeze()

                    # Only compute for single samples (batch_size=1)
                    if boundaries_pred.dim() == 1:
                        predicted_boundary_ticks, predicted_label_indices = (
                            self.model.compute_predictions(
                                output=outputs, measure_ticks=measure_ticks
                            )
                        )

                        # Ensure boundaries include start
                        if (
                            len(predicted_boundary_ticks) > 0
                            and predicted_boundary_ticks[0] != 0
                        ):
                            predicted_boundary_ticks = np.insert(
                                predicted_boundary_ticks, 0, 0
                            )
                            start_label_index = (
                                self.label_map.index("Start")
                                if "Start" in self.label_map
                                else 0
                            )
                            predicted_label_indices = np.insert(
                                predicted_label_indices, 0, start_label_index
                            )

                        # Ensure boundaries include end
                        if (
                            len(predicted_boundary_ticks) > 0
                            and predicted_boundary_ticks[-1]
                            != boundaries_pred.shape[-1] - 1
                        ):
                            predicted_boundary_ticks = np.append(
                                predicted_boundary_ticks, boundaries_pred.shape[-1] - 1
                            )
                        elif len(predicted_boundary_ticks) > 0:
                            # Last tick is the end, so final label doesn't make sense
                            predicted_label_indices = predicted_label_indices[:-1]

                        estimated_intervals = np.column_stack(
                            (
                                predicted_boundary_ticks[:-1],
                                predicted_boundary_ticks[1:],
                            )
                        )

                        if len(estimated_intervals) == 0:
                            continue

                        gt_boundary_ticks = np.where(
                            boundaries_target.cpu().numpy() > 0.5
                        )[0]
                        if len(gt_boundary_ticks) < 2:
                            continue

                        reference_intervals = np.column_stack(
                            (gt_boundary_ticks[:-1], gt_boundary_ticks[1:])
                        )

                        # Boundary detection metrics
                        boundary_prec, boundary_recall, boundary_f1 = (
                            mir_eval.segment.detection(
                                reference_intervals=reference_intervals,
                                estimated_intervals=estimated_intervals,
                            )
                        )
                        total_boundary_prec += boundary_prec
                        total_boundary_recall += boundary_recall
                        total_boundary_f1 += boundary_f1
                        num_boundary_batches += 1

                        # Pairwise metrics (requires labels)
                        if (
                            "segment_label_activations" in targets
                            and len(gt_boundary_ticks) > 1
                        ):
                            gt_label_indices = (
                                targets["segment_label_activations"]
                                .squeeze(0)
                                .cpu()
                                .numpy()[gt_boundary_ticks[:-1]]
                            )
                            gt_labels = [
                                self.label_map[idx] for idx in gt_label_indices
                            ]

                            t_max = max(
                                reference_intervals[-1, 1], estimated_intervals[-1, 1]
                            )

                            reference_intervals_adj, reference_labels = (
                                mir_eval.util.adjust_intervals(
                                    reference_intervals, gt_labels, t_min=0, t_max=t_max
                                )
                            )

                            predicted_labels = [
                                self.label_map[idx] for idx in predicted_label_indices
                            ]
                            estimated_intervals_adj, predicted_labels = (
                                mir_eval.util.adjust_intervals(
                                    estimated_intervals,
                                    predicted_labels,
                                    t_min=0,
                                    t_max=t_max,
                                )
                            )

                            if len(reference_intervals_adj) != len(
                                reference_labels
                            ) or len(estimated_intervals_adj) != len(predicted_labels):
                                continue

                            try:
                                pairwise_prec, pairwise_recall, pairwise_f1 = (
                                    mir_eval.segment.pairwise(
                                        reference_intervals=reference_intervals_adj,
                                        reference_labels=reference_labels,
                                        estimated_intervals=estimated_intervals_adj,
                                        estimated_labels=predicted_labels,
                                        frame_size=(
                                            (0.1 / 0.5) * self.cfg.target_ticks_per_beat
                                        ),
                                    )
                                )
                                total_pairwise_prec += pairwise_prec
                                total_pairwise_recall += pairwise_recall
                                total_pairwise_f1 += pairwise_f1
                            except ValueError:
                                pass

                pbar.set_postfix({
                    "batch_loss": losses["total_loss"].item(),
                    "avg_loss": total_loss / num_batches,
                })

        metrics = {"loss": total_loss / num_batches}

        if num_boundary_batches > 0:
            metrics["boundary_precision"] = total_boundary_prec / num_boundary_batches
            metrics["boundary_recall"] = total_boundary_recall / num_boundary_batches
            metrics["boundary_f1"] = total_boundary_f1 / num_boundary_batches
            metrics["pairwise_precision"] = total_pairwise_prec / num_boundary_batches
            metrics["pairwise_recall"] = total_pairwise_recall / num_boundary_batches
            metrics["pairwise_f1"] = total_pairwise_f1 / num_boundary_batches

        return metrics

    def get_val_metric_for_early_stopping(self, val_metrics: Dict[str, float]) -> float:
        """Use validation loss for early stopping."""
        return val_metrics["loss"]
