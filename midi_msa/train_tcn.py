import argparse
import gc
import json
import os
from typing import Dict, List

import mir_eval.segment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from midi_msa.data.tcn_dataset import TCNMidiDataset
from midi_msa.data.label_preprocessor import LABEL_MAP
from midi_msa.models.tcn import TCN
from midi_msa.evaluation.metrics import acc_prec_recall


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable-length sequences."""
    # Find max length in batch
    max_length = max(sample["piano_roll"].shape[-1] for sample in batch)
    
    # Pad all sequences to max length
    padded_batch = {}
    for key in batch[0].keys():
        if key == "piano_roll":
            # Pad piano rolls
            padded = []
            for sample in batch:
                piano_roll = sample[key]
                pad_length = max_length - piano_roll.shape[-1]
                padded_piano_roll = nn.functional.pad(piano_roll, (0, pad_length))
                padded.append(padded_piano_roll)
            padded_batch[key] = torch.stack(padded)
        elif key.endswith("_activation"):
            # Pad activation sequences
            padded = []
            for sample in batch:
                activation = sample[key]
                pad_length = max_length - activation.shape[-1]
                padded_activation = nn.functional.pad(activation, (0, pad_length))
                padded.append(padded_activation)
            padded_batch[key] = torch.stack(padded)
        elif key == "segment_label_activations":
            # Pad label sequences
            padded = []
            for sample in batch:
                labels = sample[key]
                pad_length = max_length - labels.shape[-1]
                padded_labels = nn.functional.pad(labels, (0, pad_length), value=-100)  # -100 for ignore index
                padded.append(padded_labels)
            padded_batch[key] = torch.stack(padded)
    
    return padded_batch


def compute_loss(
    model_output,
    targets: Dict[str, torch.Tensor],
    loss_weight_beat: float = 1.0,
    loss_weight_downbeat: float = 3.0,
    loss_weight_section: float = 10.0,
    loss_weight_function: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Compute losses for all tasks with configurable weights."""
    losses = {}
    weighted_losses = {}
    
    if "beat_activation" in targets:
        beat_loss = nn.functional.binary_cross_entropy_with_logits(
            model_output.beat_output,
            targets["beat_activation"]
        )
        losses["beat_loss"] = beat_loss
        weighted_losses["beat_loss"] = beat_loss * loss_weight_beat
    
    if "downbeat_activation" in targets:
        downbeat_loss = nn.functional.binary_cross_entropy_with_logits(
            model_output.downbeat_output,
            targets["downbeat_activation"]
        )
        losses["downbeat_loss"] = downbeat_loss
        weighted_losses["downbeat_loss"] = downbeat_loss * loss_weight_downbeat
    
    if "segment_activation" in targets:
        segment_loss = nn.functional.binary_cross_entropy_with_logits(
            model_output.segment_output,
            targets["segment_activation"]
        )
        losses["segment_loss"] = segment_loss
        weighted_losses["segment_loss"] = segment_loss * loss_weight_section
    
    if "segment_label_activations" in targets:
        _, num_classes, _ = model_output.function_outputs.shape
        function_outputs = model_output.function_outputs.permute(0, 2, 1).reshape(-1, num_classes)
        function_targets = targets["segment_label_activations"].reshape(-1)
        
        function_loss = nn.functional.cross_entropy(
            function_outputs,
            function_targets,
            ignore_index=-100
        )
        losses["function_loss"] = function_loss
        weighted_losses["function_loss"] = function_loss * loss_weight_function
    
    # Total weighted loss
    total_loss = sum(weighted_losses.values())
    losses["total_loss"] = total_loss
    
    return losses


def train_epoch(
    model: TCN,
    beat_loss_weight: float,
    downbeat_loss_weight: float,
    section_loss_weight: float,
    function_loss_weight: float,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    clip_norm: float = 1.0,
    log_wandb: bool = False
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        piano_rolls = batch["piano_roll"].to(device)
        sslm_near = batch.get("sslm_near", None)
        sslm_far = batch.get("sslm_far", None)
        if sslm_near is not None:
            sslm_near = sslm_near.to(device)
        if sslm_far is not None:
            sslm_far = sslm_far.to(device)

        targets = {k: v.to(device) for k, v in batch.items() if k != "piano_roll"}

        optimizer.zero_grad()
        outputs = model(piano_rolls, sslm_near=sslm_near, sslm_far=sslm_far)

        losses = compute_loss(outputs, targets,
                              loss_weight_beat=beat_loss_weight,
                              loss_weight_downbeat=downbeat_loss_weight,
                              loss_weight_section=section_loss_weight,
                              loss_weight_function=function_loss_weight)

        losses["total_loss"].backward()

        # Gradient clipping
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()

        # Extract loss value immediately and delete tensors to free memory
        loss_value = losses["total_loss"].item()
        total_loss += loss_value
        num_batches += 1

        progress_bar.set_postfix({
            "loss": loss_value,
            **{k: v.item() for k, v in losses.items() if k != "total_loss"}
        })

        # Log to wandb (if wandb is available)
        if log_wandb and batch_idx % 10 == 0:
            try:
                import wandb
                wandb.log({
                    "train/" + k: v.item() for k, v in losses.items()
                })
            except ImportError:
                pass

        # Clear memory every batch to prevent accumulation
        del piano_rolls, targets, outputs, losses
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    return total_loss / num_batches


def validate(
    model: TCN,
    ticks_per_beat: int,
    beat_loss_weight: float,
    downbeat_loss_weight: float,
    section_loss_weight: float,
    function_loss_weight: float,
    label_map: List[str],
    dataloader: DataLoader,
    device: torch.device,
    log_wandb: bool = False
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_losses = {}
    num_batches = 0

    total_boundary_prec = 0.0
    total_boundary_recall = 0.0
    total_boundary_f1 = 0.0

    total_pairwise_prec = 0.0
    total_pairwise_recall = 0.0
    total_pairwise_f1 = 0.0

    num_boundary_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            piano_rolls = batch["piano_roll"].to(device)
            sslm_near = batch.get("sslm_near", None)
            sslm_far = batch.get("sslm_far", None)
            if sslm_near is not None:
                sslm_near = sslm_near.to(device)
            if sslm_far is not None:
                sslm_far = sslm_far.to(device)

            targets = {k: v.to(device) for k, v in batch.items() if k != "piano_roll"}

            outputs = model(piano_rolls, sslm_near=sslm_near, sslm_far=sslm_far)
            losses = compute_loss(outputs, targets)

            measure_ticks = batch.get("measure_ticks", None)
            if measure_ticks is not None and "segment_activation" in targets:
                boundaries_pred = torch.sigmoid(outputs.segment_output).squeeze()
                boundaries_target = targets["segment_activation"].squeeze()

                # Batch dim
                if boundaries_pred.dim() == 2:  # batch_size > 1
                    print("Warning: Skipping boundary metrics for batch size > 1")
                else:  # Single sample
                    # import matplotlib.pyplot as plt
                    # plt.plot(boundaries_pred.squeeze().cpu().numpy(), label='Predicted Boundaries')
                    # plt.plot(boundaries_target.squeeze().cpu().numpy(), label='Target Boundaries')
                    # plt.legend()
                    # plt.show()
                    predicted_boundary_ticks, predicted_label_indices = model.compute_predictions(output=outputs, measure_ticks=measure_ticks)
                    estimated_intervals = np.column_stack((predicted_boundary_ticks[:-1], predicted_boundary_ticks[1:]))

                    if len(estimated_intervals) == 0:
                        continue

                    gt_boundary_ticks = np.where(boundaries_target.cpu().numpy() > 0.5)[0]
                    reference_intervals = np.column_stack((gt_boundary_ticks[:-1], gt_boundary_ticks[1:]))
                    boundary_prec, boundary_recall, boundary_f1 = mir_eval.segment.detection(
                        reference_intervals=reference_intervals,
                        estimated_intervals=estimated_intervals
                    )
                    total_boundary_prec += boundary_prec
                    total_boundary_recall += boundary_recall
                    total_boundary_f1 += boundary_f1
                    num_boundary_batches += 1

                    if "segment_label_activations" in targets and len(gt_boundary_ticks) > 1:
                        gt_label_indices = targets["segment_label_activations"].squeeze(0).cpu().numpy()[gt_boundary_ticks[:-1]]
                        gt_labels = [label_map[idx] for idx in gt_label_indices]

                        # Determine the maximum end time to ensure both intervals cover the same span
                        t_max = max(reference_intervals[-1, 1], estimated_intervals[-1, 1])

                        reference_intervals, reference_labels = mir_eval.util.adjust_intervals(
                            reference_intervals,
                            gt_labels,
                            t_min=0,
                            t_max=t_max
                        )

                        predicted_labels = [label_map[idx] for idx in predicted_label_indices]
                        estimated_intervals, predicted_labels = mir_eval.util.adjust_intervals(
                            estimated_intervals,
                            predicted_labels,
                            t_min=0,
                            t_max=t_max
                        )

                        if len(reference_intervals) != len(reference_labels) or len(estimated_intervals) != len(predicted_labels):
                            print(f"Warning: Mismatch in intervals and labels lengths. {len(reference_intervals)} != {len(reference_labels)} or {len(estimated_intervals)} != {len(predicted_labels)}")
                            continue
                        
                        # TODO: This keeps crashing in validate_structure. Check inputs.
                        try:
                            pairwise_prec, pairwise_recall, pairwise_f1 = mir_eval.segment.pairwise(
                                reference_intervals=reference_intervals,
                                reference_labels=reference_labels,
                                estimated_intervals=estimated_intervals,
                                estimated_labels=predicted_labels
                            )
                            total_pairwise_prec += pairwise_prec
                            total_pairwise_recall += pairwise_recall
                            total_pairwise_f1 += pairwise_f1
                        except ValueError as e:
                            print(f"Warning: Error computing pairwise metrics: {e}")
                            continue


            # Accumulate losses (extract values immediately)
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1

            # Clear memory every batch
            del piano_rolls, targets, outputs, losses
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    if num_boundary_batches > 0:
        avg_losses["boundary_precision"] = total_boundary_prec / num_boundary_batches
        avg_losses["boundary_recall"] = total_boundary_recall / num_boundary_batches
        avg_losses["boundary_f1"] = total_boundary_f1 / num_boundary_batches

        avg_losses["pairwise_precision"] = total_pairwise_prec / num_boundary_batches
        avg_losses["pairwise_recall"] = total_pairwise_recall / num_boundary_batches
        avg_losses["pairwise_f1"] = total_pairwise_f1 / num_boundary_batches

    if log_wandb:
        try:
            import wandb
            wandb.log({
                "val/" + k: v for k, v in avg_losses.items()
            })
        except ImportError:
            pass

    return avg_losses


def train_fold(
    fold_idx: int,
    train_midi_files: List[str],
    val_midi_files: List[str],
    args,
    label_map: List[str],
    device: torch.device
) -> Dict[str, float]:
    """Train and validate a single fold."""
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx + 1}")
    print(f"{'='*80}")

    fold_checkpoint_dir = os.path.join(args.checkpoint_dir, f"fold_{fold_idx + 1}")
    os.makedirs(fold_checkpoint_dir, exist_ok=True)

    print("Loading dataset...")
    dataset_args = {
        "midi_dir": args.midi_dir,
        "annotation_dir": args.annotation_dir,
        "piano_roll_dir": args.piano_roll_dir,
        "segment_function_vocab": label_map,
        "target_ticks_per_beat": args.target_ticks_per_beat,
        "compute_beats": False,
        "compute_downbeats": False,
        "compute_segments": True,
        "instrument_overtones": True,
        "separate_drums": True
    }
    train_dataset = TCNMidiDataset(
        midi_files=train_midi_files,
        sslms_dir=args.sslm_dir,
        **dataset_args
    )
    val_dataset = TCNMidiDataset(
        midi_files=val_midi_files,
        sslms_dir=args.sslm_dir,
        **dataset_args
    )
    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # collate_fn=collate_fn,
        pin_memory=True
    )

    input_channels = train_dataset[0]["piano_roll"].shape[0]
    model = TCN(
        input_channels=input_channels,
        segment_function_vocab=label_map,
        tcn_layers=2
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")
    best_val_metrics = {}
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model=model,
            beat_loss_weight=args.beat_loss_weight,
            downbeat_loss_weight=args.downbeat_loss_weight,
            section_loss_weight=args.section_loss_weight,
            function_loss_weight=args.function_loss_weight,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_norm=args.clip_norm,
            log_wandb=args.log_wandb
        )
        print(f"Train loss: {train_loss:.4f}")

        # Clear memory and run garbage collection between train and validation
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        val_losses = validate(
            model=model,
            ticks_per_beat=args.target_ticks_per_beat,
            beat_loss_weight=args.beat_loss_weight,
            downbeat_loss_weight=args.downbeat_loss_weight,
            section_loss_weight=args.section_loss_weight,
            function_loss_weight=args.function_loss_weight,
            label_map=label_map,
            dataloader=val_loader,
            device=device,
            log_wandb=args.log_wandb
        )
        val_loss = val_losses["total_loss"]
        print(f"Validation losses: {val_losses}")

        # Clear memory after validation
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # Save checkpoint
        if epoch % args.save_every == 0 or val_loss < best_val_loss:
            checkpoint = {
                "epoch": epoch,
                "fold": fold_idx + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "segment_vocab": label_map,
                "config": vars(args)
            }

            checkpoint_path = os.path.join(
                fold_checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save(checkpoint, checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = val_losses
                best_checkpoint_path = os.path.join(
                    fold_checkpoint_dir,
                    "best_checkpoint.pt"
                )
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Saved best checkpoint with val loss: {val_loss:.4f}")

    print(f"\nFold {fold_idx + 1} completed! Best validation loss: {best_val_loss:.4f}")
    return best_val_metrics


def main():
    parser = argparse.ArgumentParser(description="Train TCN model on MIDI data")
    parser.add_argument("--midi-dir", type=str, required=True, help="Directory containing MIDI files")
    parser.add_argument("--annotation-dir", type=str, required=True, help="Directory containing annotation files")
    parser.add_argument("--piano-roll-dir", type=str, required=True, help="Directory containing piano roll files. Will be created and populated if it doesn't exist.")
    parser.add_argument("--sslm-dir", type=str, default=None, help="Directory containing SSLM files (if using SSLM features)")
    parser.add_argument("--target-ticks-per-beat", type=int, default=48, help="Target ticks per beat")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Gradient clipping norm (set to 0 to disable)")
    # Loss weights
    parser.add_argument("--beat-loss-weight", type=float, default=1.0, help="Weight for beat loss")
    parser.add_argument("--downbeat-loss-weight", type=float, default=3.0, help="Weight for downbeat loss")
    parser.add_argument("--section-loss-weight", type=float, default=10.0, help="Weight for section boundary loss")
    parser.add_argument("--function-loss-weight", type=float, default=1.0, help="Weight for segment function loss")

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--split-file", type=str, nargs='+', default=None, help="JSON file(s) defining dataset splits. Multiple files for n-fold cross-validation.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set proportion if no split file provided")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--log-wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="midi-tcn", help="Weights & Biases project name")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    if args.log_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=args)
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            args.log_wandb = False
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Determine if we're doing cross-validation
    n_fold_cv = args.split_file and len(args.split_file) > 1
    if n_fold_cv:
        print(f"\nPerforming {len(args.split_file)}-fold cross-validation")
        print(f"Split files: {args.split_file}")

        # Load all folds
        folds = []
        for split_file in args.split_file:
            with open(split_file, "r") as f:
                splits = json.load(f)
                folds.append({
                    "train": splits.get("train", []),
                    "val": splits.get("val", [])
                })

        label_map = list(set(LABEL_MAP.values()))
        print(f"Segment function vocab: {label_map}")

        device = torch.device(args.device)

        # Train each fold and collect results
        fold_results = []
        for fold_idx, fold in enumerate(folds):
            best_metrics = train_fold(
                fold_idx=fold_idx,
                train_midi_files=fold["train"],
                val_midi_files=fold["val"],
                args=args,
                label_map=label_map,
                device=device
            )
            fold_results.append(best_metrics)

            # Clear memory between folds
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        # Aggregate results across folds
        print(f"\n{'='*80}")
        print("Cross-Validation Results")
        print(f"{'='*80}")

        # Compute mean and std for each metric
        all_metrics = {}
        for metric_name in fold_results[0].keys():
            values = [fold[metric_name] for fold in fold_results]
            mean_val = sum(values) / len(values)
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            all_metrics[metric_name] = {
                "mean": mean_val,
                "std": std_val,
                "values": values
            }

        # Summary
        for metric_name, stats in all_metrics.items():
            print(f"{metric_name}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Folds: {[f'{v:.4f}' for v in stats['values']]}")

        cv_results_path = os.path.join(args.checkpoint_dir, "cv_results.json")
        with open(cv_results_path, "w") as f:
            json.dump({
                "n_folds": len(folds),
                "metrics": all_metrics,
                "fold_results": fold_results
            }, f, indent=2)
        print(f"\nCross-validation results saved to: {cv_results_path}")

    else:
        if args.split_file:
            print("Using single split file")
            with open(args.split_file[0], "r") as f:
                splits = json.load(f)
                train_midi_files = splits.get("train", [])
                val_midi_files = splits.get("val", [])
        else:
            # Recursively collect all MIDI files in subdirectories
            all_midi_files = []
            for _, _, files in os.walk(args.midi_dir):
                for filename in files:
                    if (filename.endswith('.mid') or filename.endswith('.midi')) and not filename.startswith('.'):
                        file_id, _ = os.path.splitext(filename)
                        all_midi_files.append(file_id)
            total_size = len(all_midi_files)
            val_size = int(total_size * args.val_split)
            train_size = total_size - val_size

            train_midi_files = all_midi_files[:train_size]
            val_midi_files = all_midi_files[train_size:train_size + val_size]

        label_map = list(set(LABEL_MAP.values()))
        print(f"Segment function vocab: {label_map}")

        device = torch.device(args.device)

        train_fold(
            fold_idx=0,
            train_midi_files=train_midi_files,
            val_midi_files=val_midi_files,
            args=args,
            label_map=label_map,
            device=device
        )

    print("\nTraining completed!")
    
    if args.log_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()