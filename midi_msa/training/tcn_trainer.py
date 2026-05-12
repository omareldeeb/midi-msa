import glob
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.label_preprocessor import LABEL_MAP_TRAIN, LABEL_MAP_VAL
from ..data.tcn_dataset import TCNMidiDataset
from ..evaluation.tcn_evaluation import validate_tcn_model
from .base_trainer import BaseTrainer


class TCNTrainer(BaseTrainer):
    """Trainer for TCN (sequence-based) method."""

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)

        # Build segment vocabulary
        self.label_map_train = LABEL_MAP_TRAIN
        self.label_map_val = LABEL_MAP_VAL

        self.segment_function_vocab_train = sorted(list(set(self.label_map_train.values())))
        self.segment_function_vocab_val = sorted(list(set(self.label_map_val.values())))

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
                str(Path(self.cfg.annotation_dir) / "*_functions_qn.json")
            )
            np.random.shuffle(all_files)

            file_ids = [
                Path(f).stem.replace("_functions_qn", "") for f in all_files
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
            "extra_midi_dir": self.cfg.extra_midi_dir,
            "piano_roll_dir": self.cfg.piano_roll_dir,
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
            segment_function_vocab=self.segment_function_vocab_train,
            label_map=self.label_map_train,
            **dataset_args
        )
        val_dataset = TCNMidiDataset(
            midi_files=val_files, sslm_dir=self.cfg.sslm_dir,
            transpose_augmentation=False,
            segment_function_vocab=self.segment_function_vocab_val,
            label_map=self.label_map_val,
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
        returns: (num_segments, D)
        """
        pooled = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else embeddings.shape[-1]
            if end > start:
                pooled.append(embeddings[:, start: end].mean(dim=1))
        return torch.stack(pooled)

    def nt_xent_loss(self, embeddings, boundaries, labels, temperature=0.1):
        assert embeddings.shape[0] == 1, 'nt_xent_loss only implemented for batch_size == 1'
        embeddings = embeddings.squeeze(0)  # (D, T)
        segment_embeddings = self.pool_segments(embeddings, boundaries)  # (#segments, D)
        z = segment_embeddings

        z = nn.functional.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / temperature  # (S, S)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max  # numerical stability

        labels = torch.tensor([self.segment_function_vocab_train.index(x) for x in labels], device=sim.device)
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # positives

        # remove self-comparisons
        diag = torch.eye(sim.size(0), device=sim.device).bool()
        mask = mask & ~diag

        exp_sim = torch.exp(sim)
        exp_sim = exp_sim * (~diag)
        denom = exp_sim.sum(dim=1)
        pos = exp_sim * mask
        valid_rows = pos.sum(dim=1) > 0
        if valid_rows.sum() == 0:
            return segment_embeddings.sum() * 0.0
            # return torch.tensor(0.0, device=segment_embeddings.device)

        # old log of sum loss
        old_loss = -torch.log((pos.sum(dim=1)[valid_rows] + 1e-8) / denom[valid_rows])

        # sum of logs loss (SupCon loss: https://arxiv.org/pdf/2004.11362 eq 2)
        diag = torch.eye(sim.size(0), device=sim.device).bool()
        sim_masked = sim.masked_fill(diag, float('-inf'))
        log_denom = torch.logsumexp(sim_masked, dim=1)
        log_probs = sim - log_denom.unsqueeze(1)  # sim is the log of exp_sim, so log_probs = log(exp_sim/denom), where denom = sum of exp(sim)
        pos_log_probs = log_probs * mask
        pos_sum = pos_log_probs.sum(dim=1)
        pos_counts = mask.sum(dim=1)
        loss = -pos_sum[valid_rows] / pos_counts[valid_rows]

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
            if self.cfg.function_output_activation == "sigmoid":
                valid_mask = function_targets != -100
                if valid_mask.any():
                    function_target_multi = torch.zeros_like(function_outputs)
                    function_target_multi[valid_mask, function_targets[valid_mask].long()] = 1.0
                    raw_loss = nn.functional.binary_cross_entropy_with_logits(
                        function_outputs,
                        function_target_multi,
                        reduction="none",
                    )
                    function_loss = (
                        raw_loss * valid_mask.unsqueeze(1)
                    ).sum() / (valid_mask.sum() * num_classes)
                else:
                    function_loss = function_outputs.sum() * 0.0
            else:
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

    def validate_epoch(self, val_loaders, boundary_f1_discard_first_and_last=True) -> Dict[str, float]:
        """Validate for one epoch."""
        return validate_tcn_model(model=self.model,
                                  val_loader=val_loaders[0],
                                  label_map_train=self.label_map_train,
                                  segment_vocab_train=self.segment_function_vocab_train,
                                  label_map_val=self.label_map_val,
                                  segment_vocab_val=self.segment_function_vocab_val,
                                  device=self.device,
                                  loss_fn=self.compute_loss,
                                  boundary_f1_discard_first_and_last=boundary_f1_discard_first_and_last,
                                  function_activation=self.cfg.function_output_activation,
                                  )

    def get_val_metric_for_early_stopping(self, val_metrics: Dict[str, float], epoch: int) -> float:
        """Use validation loss for early stopping."""
        multiplier = min(1.0, epoch / 29)  # penalize early epochs
        return val_metrics["primary_optimization_metric"] * multiplier
