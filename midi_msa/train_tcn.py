import argparse
import json
import os
from typing import Dict, List

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
    loss_weight_section: float = 15.0,
    loss_weight_function: float = 0.1
) -> Dict[str, torch.Tensor]:
    """Compute losses for all tasks with configurable weights."""
    losses = {}
    weighted_losses = {}
    
    # Beat detection loss
    if "beat_activation" in targets:
        beat_loss = nn.functional.binary_cross_entropy_with_logits(
            model_output.beat_output,
            targets["beat_activation"]
        )
        losses["beat_loss"] = beat_loss
        weighted_losses["beat_loss"] = beat_loss * loss_weight_beat
    
    # Downbeat detection loss
    if "downbeat_activation" in targets:
        downbeat_loss = nn.functional.binary_cross_entropy_with_logits(
            model_output.downbeat_output,
            targets["downbeat_activation"]
        )
        losses["downbeat_loss"] = downbeat_loss
        weighted_losses["downbeat_loss"] = downbeat_loss * loss_weight_downbeat
    
    # Segment boundary detection loss
    if "segment_activation" in targets:
        segment_loss = nn.functional.binary_cross_entropy_with_logits(
            model_output.segment_output,
            targets["segment_activation"]
        )
        losses["segment_loss"] = segment_loss
        weighted_losses["segment_loss"] = segment_loss * loss_weight_section
    
    # Segment function classification loss
    if "segment_label_activations" in targets:
        # Reshape outputs and targets
        _, num_classes, _ = model_output.function_outputs.shape
        function_outputs = model_output.function_outputs.permute(0, 2, 1).reshape(-1, num_classes)
        function_targets = targets["segment_label_activations"].reshape(-1)
        
        # Compute cross entropy loss, ignoring padded positions (-100)
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
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_wandb: bool = False
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        piano_rolls = batch["piano_roll"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "piano_roll"}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(piano_rolls)
        
        # Compute losses
        losses = compute_loss(outputs, targets)
        
        # Backward pass
        losses["total_loss"].backward()
        optimizer.step()
        
        # Update metrics
        total_loss += losses["total_loss"].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": losses["total_loss"].item(),
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
    
    return total_loss / num_batches


def validate(
    model: TCN,
    dataloader: DataLoader,
    device: torch.device,
    log_wandb: bool = False
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_losses = {}
    num_batches = 0
    
    # Initialize boundary metrics accumulators
    total_boundary_acc = 0.0
    total_boundary_prec = 0.0
    total_boundary_recall = 0.0
    num_boundary_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            piano_rolls = batch["piano_roll"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "piano_roll"}
            
            outputs = model(piano_rolls)
            losses = compute_loss(outputs, targets)

            # Compute boundary metrics if measure_ticks are available
            measure_ticks = batch.get("measure_ticks", None)
            if measure_ticks is not None and "segment_activation" in targets:
                # Get predictions and targets for boundary positions
                boundaries_pred = torch.sigmoid(outputs.segment_output).squeeze()
                boundaries_target = targets["segment_activation"].squeeze()
                
                # Handle batch dimension properly
                if boundaries_pred.dim() == 2:  # batch_size > 1
                    # Process each sample in the batch
                    batch_acc = []
                    batch_prec = []
                    batch_recall = []
                    for i in range(boundaries_pred.shape[0]):
                        acc, prec, recall = acc_prec_recall(
                            boundaries_pred[i].cpu()[measure_ticks[i]],
                            boundaries_target[i].cpu()[measure_ticks[i]]
                        )
                        batch_acc.append(acc)
                        batch_prec.append(prec)
                        batch_recall.append(recall)
                    
                    if batch_acc:  # If we have valid metrics
                        total_boundary_acc += sum(batch_acc) / len(batch_acc)
                        total_boundary_prec += sum(batch_prec) / len(batch_prec)
                        total_boundary_recall += sum(batch_recall) / len(batch_recall)
                        num_boundary_batches += 1
                else:  # Single sample
                    boundary_acc, boundary_prec, boundary_recall = acc_prec_recall(
                        boundaries_pred.cpu()[measure_ticks],
                        boundaries_target.cpu()[measure_ticks]
                    )
                    total_boundary_acc += boundary_acc
                    total_boundary_prec += boundary_prec
                    total_boundary_recall += boundary_recall
                    num_boundary_batches += 1
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Add averaged boundary metrics
    if num_boundary_batches > 0:
        avg_losses["boundary_acc"] = total_boundary_acc / num_boundary_batches
        avg_losses["boundary_prec"] = total_boundary_prec / num_boundary_batches
        avg_losses["boundary_recall"] = total_boundary_recall / num_boundary_batches
    
    # Log to wandb (if wandb is available)
    if log_wandb:
        try:
            import wandb
            wandb.log({
                "val/" + k: v for k, v in avg_losses.items()
            })
        except ImportError:
            pass
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="Train TCN model on MIDI data")
    parser.add_argument("--midi-dir", type=str, required=True, help="Directory containing MIDI files")
    parser.add_argument("--annotation-dir", type=str, required=True, help="Directory containing annotation files")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--split-file", type=str, default=None, help="JSON file defining dataset splits")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set proportion if no split file provided")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--log-wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="midi-tcn", help="Weights & Biases project name")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Initialize wandb (if available)
    if args.log_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=args)
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            args.log_wandb = False
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.split_file:
        print("got split file")
        with open(args.split_file, "r") as f:
            splits = json.load(f)
            train_midi_files = splits.get("train", [])
            val_midi_files = splits.get("val", [])
    else:
        # Recursively collect all MIDI files in subdirectories
        all_midi_files = []
        for root, _, files in os.walk(args.midi_dir):
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

    # Load dataset
    print("Loading dataset...")
    dataset_args = {
        "midi_dir": args.midi_dir,
        "annotation_dir": args.annotation_dir,
        "segment_function_vocab": label_map,
        "target_ticks_per_beat": 4,
        "compute_beats": False,
        "compute_downbeats": False,
        "compute_segments": True,
        "instrument_overtones": True,
        "separate_drums": True
    }
    train_dataset = TCNMidiDataset(
        midi_files=train_midi_files,
        **dataset_args
    )

    val_dataset = TCNMidiDataset(
        midi_files=val_midi_files,
        **dataset_args
    )

    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Segment function vocab: {label_map}")

    # Create data loaders
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
    
    # Initialize model
    input_channels = train_dataset[0]["piano_roll"].shape[0]
    print(f"Input channels: {input_channels}")
    device = torch.device(args.device)
    model = TCN(
        input_channels=input_channels,
        segment_function_vocab=label_map,
        tcn_layers=2
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, args.log_wandb
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_losses = validate(model, val_loader, device, args.log_wandb)
        val_loss = val_losses["total_loss"]
        print(f"Validation losses: {val_losses}")
        
        # Save checkpoint
        if epoch % args.save_every == 0 or val_loss < best_val_loss:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "segment_vocab": label_map,
                "config": vars(args)
            }
            
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    "best_checkpoint.pt"
                )
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Saved best checkpoint with val loss: {val_loss:.4f}")
    
    print("\nTraining completed!")
    
    if args.log_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()