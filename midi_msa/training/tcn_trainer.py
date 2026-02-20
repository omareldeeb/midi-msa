import glob
import json
from pathlib import Path
from typing import Dict, Tuple

import mir_eval.segment
import mir_eval.util
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm

from ..data import utils
from ..data.label_preprocessor import LABEL_MAP
from ..data.tcn_dataset import TCNMidiDataset
from .base_trainer import BaseTrainer


class TCNTrainer(BaseTrainer):
    """Trainer for TCN (sequence-based) method."""

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)

        # Build segment vocabulary
        self.label_map = sorted(list(set(LABEL_MAP.values())))

    def lower_is_better(self) -> bool:
        return False  # validation metric should be maximized

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
            "compute_segment_labels": self.cfg.compute_segment_labels,
            "instrument_overtones": self.cfg.instrument_overtones,
            "separate_drums": self.cfg.separate_drums,
        }

        train_dataset = TCNMidiDataset(
            midi_files=train_files, sslm_dir=self.cfg.sslm_dir,
            transpose_augmentation=self.cfg.transpose_augmentation,
            **dataset_args
        )
        val_dataset = TCNMidiDataset(
            midi_files=val_files, sslm_dir=self.cfg.sslm_dir,
            transpose_augmentation=False,
            **dataset_args
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

    def pool_segments(self, embeddings, boundaries):
        """
        embeddings: (D, T)
        boundaries: list[int] sorted, including 0
        returns: (D, num_segments)
        """
        pooled = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else embeddings.shape[-1]
            if end > start:
                pooled.append(embeddings[:, start: end].mean(dim=1))
        return torch.stack(pooled)

    def nt_xent_loss(self, embeddings, boundaries, labels, temperature=0.1):
        embeddings = embeddings.squeeze(0)  # (D, T)
        segment_embeddings = self.pool_segments(embeddings, boundaries)  # (D, #segments)
        z = segment_embeddings

        z = nn.functional.normalize(z, dim=0)
        sim = torch.matmul(z, z.T) / temperature  # (S, S)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max  # numerical stability

        labels = torch.tensor([self.label_map.index(x) for x in labels], device=sim.device)
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # positives
        # pos_counts = mask.sum(dim=1)
        # valid = pos_counts > 0
        # if valid.sum() == 0:
        #     return torch.tensor(0, device=embeddings.device, requires_grad=True)

        # remove self-comparisons
        diag = torch.eye(sim.size(0), device=sim.device).bool()
        mask = mask & ~diag

        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1)
        pos = exp_sim * mask
        valid_rows = pos.sum(dim=1) > 0
        if valid_rows.sum() == 0:
            return torch.tensor(0.0, device=segment_embeddings.device)

        loss = -torch.log((pos.sum(dim=1)[valid_rows] + 1e-8) / denom[valid_rows])

        return loss.mean()

    def compute_loss(self, model_output, targets: Dict[str, torch.Tensor], batch) -> Dict:
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

            if self.cfg.contrastive_loss_weight > 0:
                losses['contrastive_loss'] = self.nt_xent_loss(embeddings=model_output.segment_embeddings,
                                                      boundaries=[int(x) for x in batch.get('segment_ticks_in_piano_roll')],
                                                      labels=[str(x[0]) for x in batch.get('segment_labels_in_piano_roll')])
                weighted_losses['contrastive_loss'] = self.cfg.contrastive_loss_weight * losses.get('contrastive_loss')
            else:
                losses['contrastive_loss'] = 0.0
                weighted_losses['contrastive_loss'] = 0.0

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
            piano_rolls = batch["piano_roll"].to(torch.float32).to(self.device)
            sslm_near = batch.get("sslm_near")
            sslm_far = batch.get("sslm_far")

            if sslm_near is not None:
                sslm_near = sslm_near.to(torch.float32).to(self.device)
            if sslm_far is not None:
                sslm_far = sslm_far.to(torch.float32).to(self.device)

            targets = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k not in ["piano_roll", "sslm_near", "sslm_far", "measure_ticks", 'file_id',
                             'segment_ticks_in_piano_roll', 'segment_labels_in_piano_roll']
            }

            self.optimizer.zero_grad()
            outputs = self.model(piano_rolls, sslm_near=sslm_near, sslm_far=sslm_far)

            losses = self.compute_loss(outputs, targets, batch)

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
        total_beat_f1 = 0.0
        total_downbeat_f1 = 0.0
        total_boundary_prec = 0.0
        total_boundary_recall = 0.0
        total_boundary_f1 = 0.0
        total_pairwise_prec = 0.0
        total_pairwise_recall = 0.0
        total_pairwise_f1 = 0.0
        total_label_f1 = {label: 0.0 for label in self.label_map}
        total_label_accuracy = 0.0
        num_boundary_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                piano_rolls = batch["piano_roll"].to(torch.float32).to(self.device)
                sslm_near = batch.get("sslm_near")
                sslm_far = batch.get("sslm_far")
                measure_ticks = batch.get("measure_ticks")

                if sslm_near is not None:
                    sslm_near = sslm_near.to(torch.float32).to(self.device)
                if sslm_far is not None:
                    sslm_far = sslm_far.to(torch.float32).to(self.device)

                targets = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k not in ["piano_roll", "sslm_near", "sslm_far", "measure_ticks", 'file_id',
                                 'segment_ticks_in_piano_roll', 'segment_labels_in_piano_roll']
                }

                outputs = self.model(
                    piano_rolls, sslm_near=sslm_near, sslm_far=sslm_far
                )
                losses = self.compute_loss(outputs, targets, batch)

                total_loss += losses["total_loss"].item()
                num_batches += 1
                t_max = piano_rolls.shape[-1] - 1

                # Compute F1 for function labels
                # if outputs.function_outputs is not None and "segment_label_activations" in targets:
                #     predicted_label_probabilities = torch.softmax(
                #         outputs.function_outputs.squeeze(), dim=-2
                #     ).argmax(dim=-2)
                #
                #     true_labels = targets["segment_label_activations"].squeeze()
                #
                #     predicted_label_probabilities_flat = predicted_label_probabilities.view(-1)
                #     true_labels_flat = true_labels.view(-1)
                #
                #     f1 = multiclass_f1_score(
                #         predicted_label_probabilities_flat,
                #         true_labels_flat,
                #         num_classes=len(self.label_map),
                #         average=None,
                #     )
                #
                #     unique_labels = torch.unique(true_labels_flat)
                #     for label_idx, label in enumerate(self.label_map):
                #         if label_idx in unique_labels:
                #             label_f1 = f1[label_idx].item()
                #             total_label_f1[label] += label_f1

                # Compute beat f1
                # Only compute for single samples (batch_size=1)
                if 'beat_activation' in targets and targets['beat_activation'].shape[0] == 1:
                    true_beats = torch.where(targets['beat_activation'].squeeze() == 1.0)[0]
                    predicted_beats = utils.extract_peaks(outputs.beat_output.squeeze())
                    relevant = set(x.item() for x in true_beats)
                    retrieved = set(x.item() for x in predicted_beats)
                    beat_f1 = utils.generic_F1(numerator=len(relevant.intersection(retrieved)),
                                               n_relevant=len(relevant),
                                               n_retrieved=len(retrieved))
                    total_beat_f1 += beat_f1

                # Compute downbeat f1
                # Only compute for single samples (batch_size=1)
                if 'downbeat_activation' in targets and targets['downbeat_activation'].shape[0] == 1:
                    true_downbeats = torch.where(targets['downbeat_activation'].squeeze() == 1.0)[0]
                    predicted_downbeats = utils.extract_peaks(outputs.downbeat_output.squeeze())
                    relevant = set(x.item() for x in true_downbeats)
                    retrieved = set(x.item() for x in predicted_downbeats)
                    downbeat_f1 = utils.generic_F1(numerator=len(relevant.intersection(retrieved)),
                                                   n_relevant=len(relevant),
                                                   n_retrieved=len(retrieved))
                    total_downbeat_f1 += downbeat_f1

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

                        # add one tick beyond the end for stacking purposes
                        predicted_boundary_ticks = [int(x) for x in predicted_boundary_ticks]
                        if predicted_boundary_ticks and predicted_boundary_ticks[-1] != t_max + 1:
                            predicted_boundary_ticks.append(t_max + 1)
                        estimated_intervals = np.column_stack(
                            (
                                predicted_boundary_ticks[:-1],
                                predicted_boundary_ticks[1:],
                            )
                        )

                        if len(estimated_intervals) == 0:
                            continue

                        gt_boundary_ticks = [int(x) for x in batch.get('segment_ticks_in_piano_roll')]
                        # add one tick beyond the end for stacking purposes
                        if gt_boundary_ticks and gt_boundary_ticks[-1] != t_max + 1:
                            gt_boundary_ticks.append(t_max + 1)

                        if len(gt_boundary_ticks) < 2:
                            continue

                        reference_intervals = np.column_stack(
                            (gt_boundary_ticks[:-1], gt_boundary_ticks[1:])
                        )

                        try:
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

                        except Exception as e:
                            print(f'Exception computing boundary metrics, {e}')
                            print('estimated intervals:', estimated_intervals)
                            print('ref int:', reference_intervals)
                            print('file', batch['file_id'])


                        # Pairwise metrics (requires labels)
                        if (
                            "segment_label_activations" in targets
                            and len(gt_boundary_ticks) > 1
                        ):

                            gt_labels = [str(x[0]) for x in batch.get('segment_labels_in_piano_roll')]

                            # Shouldn't need to do this?
                            reference_intervals_adj, reference_labels = (
                                mir_eval.util.adjust_intervals(
                                    reference_intervals, gt_labels, t_min=0, t_max=t_max + 1
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
                                    t_max=t_max + 1,
                                )
                            )

                            assert reference_intervals_adj.shape == reference_intervals.shape
                            assert (reference_intervals_adj == reference_intervals).all()
                            # if reference_intervals_adj.shape != reference_intervals.shape or not (reference_intervals_adj == reference_intervals).all():
                            #     print('yikes')
                            #     print('file', batch['file_id'])
                            #     print(reference_intervals_adj)
                            #     print(reference_intervals)
                            assert estimated_intervals_adj.shape == estimated_intervals.shape
                            assert (estimated_intervals_adj == estimated_intervals).all()
                            # if estimated_intervals_adj.shape != estimated_intervals.shape or not (estimated_intervals_adj == estimated_intervals).all():
                            #     print('yikes 2')
                            #     print('file', batch['file_id'])
                            #     print(estimated_intervals_adj)
                            #     print(estimated_intervals)
                            #     print(reference_intervals_adj)
                            #     print(reference_intervals)
                            #     print(gt_labels)
                            assert len(reference_intervals_adj) == len(reference_labels)
                            assert len(estimated_intervals_adj) == len(predicted_labels)
                            # if len(reference_intervals_adj) != len(reference_labels) or len(estimated_intervals_adj) != len(predicted_labels):
                            #     print('yikes3')
                            #     print('file', batch['file_id'])
                            #     print(len(reference_intervals_adj), len(reference_labels))
                            #     print(reference_intervals_adj)
                            #     print(reference_labels)
                            #     print(len(estimated_intervals_adj), len(predicted_labels))
                            #     continue

                            try:
                                pairwise_prec, pairwise_recall, pairwise_f1 = (
                                    mir_eval.segment.pairwise(
                                        reference_intervals=reference_intervals_adj,
                                        reference_labels=reference_labels,
                                        estimated_intervals=estimated_intervals_adj,
                                        estimated_labels=predicted_labels,
                                        frame_size=(
                                            # (0.1 / 0.5) * self.cfg.target_ticks_per_beat
                                            1.0
                                        ),
                                    )
                                )
                                total_pairwise_prec += pairwise_prec
                                total_pairwise_recall += pairwise_recall
                                total_pairwise_f1 += pairwise_f1
                            except ValueError as err:
                                print(f'Warning: Error in mir_eval.segment.pairwise: {err}')

                            # compute tick-wise label accuracy
                            true_segment_label_idxs = targets['segment_label_activations'][0]
                            predicted_label_idxs = torch.argmax(outputs.function_outputs[0], dim=0)
                            accuracy = sum(true_segment_label_idxs == predicted_label_idxs) / piano_rolls.shape[-1]
                            accuracy = accuracy.item()
                            total_label_accuracy += accuracy


                pbar.set_postfix({
                    "batch_loss": losses["total_loss"].item(),
                    "avg_loss": total_loss / num_batches,
                })

        metrics = {"loss": total_loss / num_batches}

        if num_boundary_batches > 0:
            metrics['beat_f1'] = total_beat_f1 / num_boundary_batches
            metrics['downbeat_f1'] = total_downbeat_f1 / num_boundary_batches
            metrics["boundary_precision"] = total_boundary_prec / num_boundary_batches
            metrics["boundary_recall"] = total_boundary_recall / num_boundary_batches
            metrics["boundary_f1"] = total_boundary_f1 / num_boundary_batches
            metrics["pairwise_precision"] = total_pairwise_prec / num_boundary_batches
            metrics["pairwise_recall"] = total_pairwise_recall / num_boundary_batches
            metrics["pairwise_f1"] = total_pairwise_f1 / num_boundary_batches
            # for label in self.label_map:
            #     metrics[f"f1_{label}"] = total_label_f1[label] / num_boundary_batches
            # # Average label F1
            # metrics["average_label_f1"] = np.mean([total_label_f1[label] / num_boundary_batches for label in self.label_map])
            metrics['label_accuracy'] = total_label_accuracy / num_boundary_batches
            metrics['primary_optimization_metric'] = (metrics['boundary_f1'] + metrics['pairwise_f1']) / 2

        return metrics

    def get_val_metric_for_early_stopping(self, val_metrics: Dict[str, float]) -> float:
        """Use validation loss for early stopping."""
        return val_metrics["primary_optimization_metric"]
