import argparse
import json
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from midi_msa.data.piano_roll_dataset import PianoRollDataset
from midi_msa.data.utils import get_piano_roll_patches
from midi_msa.models.mobilenet_boundary_classifier import MobileNetBoundaryClassifier as BoundaryClassifier
from midi_msa.evaluation.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a boundary classifier on piano-roll patches"
    )
    parser.add_argument("--window-half-ticks", type=int, default=256,
                        help="Half window size in ticks for each patch")
    parser.add_argument("--pretrained", action="store_true",
                        help="Whether to use pre-trained weights for mobilenet")
    parser.add_argument("--instrument-overtones", action="store_true",
                        help="Whether to use instrument overtone encoding")
    parser.add_argument("--patch-normalize", action="store_true",
                        help="Apply patch normalization")
    parser.add_argument("--separate-drums", action="store_true",
                        help="Train separate drum patches")
    parser.add_argument("--num-targets", type=int, default=1,
                        help="Number of target classes")
    parser.add_argument("--drop-boundary-patches", action="store_true"
                        help="Pad boundary patches to full window size")

    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training and validation")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Maximum number of epochs to train")
    parser.add_argument("--positive-oversampling-factor", type=int, default=2,
                        help="Oversample positive patches each epoch")
    parser.add_argument("--negative-undersampling-factor", type=int, default=1,
                        help="Resample negative patches each epoch")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization) coefficient")

    parser.add_argument("--data-dir", type=str,
                        help="Directory containing the piano roll patches and metadata as saved by `create_lakh_dataset` in utils.py")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint if available")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Directory to save and load model checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (e.g., 'cuda', 'cpu'). Auto-detect if not given.")
    return parser.parse_args()


def get_dataloaders(
    data_dir: Union[Path, str],
    window_half_ticks: int,
    positive_oversampling_factor: int,
    negative_undersampling_factor: int,
    pad_boundary_patches: bool,
    batch_size: int,
    patch_normalize: bool
):
    piano_rolls, patch_data = get_piano_roll_patches(
        data_dir=data_dir,
        window_half_ticks=window_half_ticks,
        positive_oversampling_factor=positive_oversampling_factor,
        negative_undersampling_factor=negative_undersampling_factor,
        pad_boundary_patches=pad_boundary_patches
    )
    metadata_df = pd.DataFrame.from_dict(patch_data, orient='index').sample(frac=1)
    
    metadata_df = metadata_df.sample(frac=1)
    metadata_train = metadata_df[metadata_df["key"].isin(["tubb_train", "non_tubb_train"])]
    metadata_val_tubb = metadata_df[metadata_df["key"] == "tubb_val"]
    metadata_val_non_tubb = metadata_df[metadata_df["key"] == "non_tubb_val"]
    metadata_train.reset_index(drop=True, inplace=True)
    metadata_val_tubb.reset_index(drop=True, inplace=True)
    metadata_val_non_tubb.reset_index(drop=True, inplace=True)

    dataset_train = PianoRollDataset(piano_rolls, metadata_train, normalize=patch_normalize)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val_tubb = PianoRollDataset(piano_rolls, metadata_val_tubb, normalize=patch_normalize)
    dataloader_val_tubb = DataLoader(dataset_val_tubb, batch_size=batch_size, shuffle=False)

    dataset_val_non_tubb = PianoRollDataset(piano_rolls, metadata_val_non_tubb, normalize=patch_normalize)
    dataloader_val_non_tubb = DataLoader(dataset_val_non_tubb, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val_tubb, dataloader_val_non_tubb


def main():
    args = parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    writer = SummaryWriter(log_dir=args.log_dir)

    # Prepare checkpoint path
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = (
        f"pretrain_{args.pretrained}_"
        f"mn_overtones_{args.instrument_overtones}_"
        f"normalized_{int(args.patch_normalize)}_"
        f"separate_drums_{int(args.separate_drums)}_"
        f"targets_{args.num_targets}.pt"
    )
    checkpoint_path = model_dir / model_name

    model = BoundaryClassifier(num_targets=args.num_targets, pretrained=pretrained).to(device)
    if args.resume and checkpoint_path.exists():
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # History storage
    history = {
        'train_loss': [],
        'val_loss_tubb': [], 'val_acc_tubb': [], 'val_prec_tubb': [], 'val_rec_tubb': [],
        'val_loss_non_tubb': [], 'val_acc_non_tubb': [], 'val_prec_non_tubb': [], 'val_rec_non_tubb': []
    }

    best_val_f1 = 0.0
    epochs_no_improve = 0

    train_dataloader, val_dataloader_tubb, val_dataloader_non_tubb = get_dataloaders(
        data_dir=args.data_dir,
        window_half_ticks=args.window_half_ticks,
        positive_oversampling_factor=args.positive_oversampling_factor,
        negative_undersampling_factor=args.negative_undersampling_factor,
        pad_boundary_patches=args.drop_boundary_patches,
        batch_size=args.batch_size,
        patch_normalize=args.patch_normalize
    )

    for epoch in range(args.num_epochs):
        # Reload piano rolls and reinitialize dataloaders if we have negative undersampling in order to reshuffle
        if args.negative_undersampling_factor and epoch > 0:
            print("Reloading data for next epoch with new sampling")
            train_dataloader, val_dataloader_tubb, val_dataloader_non_tubb = get_dataloaders(
                data_dir=args.data_dir,
                window_half_ticks=args.window_half_ticks,
                positive_oversampling_factor=args.positive_oversampling_factor,
                negative_undersampling_factor=args.negative_undersampling_factor,
                pad_boundary_patches=args.drop_boundary_patches,
                batch_size=args.batch_size,
                patch_normalize=args.patch_normalize
            )

        # Log example images
        imgs, targets = next(iter(train_dataloader))
        for i in range(min(4, len(imgs))):
            writer.add_image(
                tag=f"Train/Example_{i}_Label_{int(targets[i])}",
                img_tensor=imgs[i],
                global_step=epoch,
                dataformats="CHW",
            )

        # Training step
        model.train()
        total_loss = 0.0
        step = 0
        for batch in (pbar := tqdm(train_dataloader)):
            pbar.set_description(f"Epoch {epoch + 1}/{args.num_epochs} - Training")
            piano_roll, targets = batch
            piano_roll, targets = piano_roll.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(piano_roll)
            loss = criterion(output, targets.float().to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Train Loss (current/average)": f"{loss.item():.4f}/{total_loss / (step + 1):.4f}"})

            writer.add_scalar("Loss/Train", loss, epoch * len(train_dataloader) + step)

            step += 1
        avg_train_loss = total_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        writer.flush()

        # Validation step
        model.eval()
        val_outputs_tubb, val_targets_tubb = [], []
        val_outputs_non_tubb, val_targets_non_tubb = [], []
        val_loss_tubb, val_loss_non_tubb = 0, 0
        with torch.no_grad():
            for batch_tubb in val_dataloader_tubb:
                piano_roll, targets = batch_tubb
                piano_roll, targets = piano_roll.to(device), targets.to(device)

                output = model(piano_roll)

                val_outputs_tubb.append(output)
                val_targets_tubb.append(targets)

                loss = criterion(output, targets.float().to(device))
                val_loss_tubb += loss.item()
            for batch_non_tubb in val_dataloader_non_tubb:
                piano_roll, targets = batch_non_tubb
                piano_roll, targets = piano_roll.to(device), targets.to(device)

                output = model(piano_roll)

                val_outputs_non_tubb.append(output)
                val_targets_non_tubb.append(targets)

                loss = criterion(output, targets.float().to(device))
                val_loss_non_tubb += loss.item()

            val_loss_tubb /= len(val_dataloader_tubb)
            val_loss_non_tubb /= len(val_dataloader_non_tubb)

        metrics_tubb = compute_metrics(
            torch.cat(val_outputs_tubb), torch.cat(val_targets_tubb)
        )
        metrics_non_tubb = compute_metrics(
            torch.cat(val_outputs_non_tubb), torch.cat(val_targets_non_tubb)
        )

        history['val_loss_tubb'].append(val_loss_tubb)
        history['val_acc_tubb'].append(metrics_tubb['accuracy_0'])
        history['val_prec_tubb'].append(metrics_tubb['precision_0'])
        history['val_rec_tubb'].append(metrics_tubb['recall_0'])
        history['val_loss_non_tubb'].append(val_loss_non_tubb)
        history['val_acc_non_tubb'].append(metrics_non_tubb['accuracy_0'])
        history['val_prec_non_tubb'].append(metrics_non_tubb['precision_0'])
        history['val_rec_non_tubb'].append(metrics_non_tubb['recall_0'])

        f1_t = 2 * (metrics_tubb['precision_0'] * metrics_tubb['recall_0']) / (
            metrics_tubb['precision_0'] + metrics_tubb['recall_0'] + 1e-8
        )
        f1_n = 2 * (metrics_non_tubb['precision_0'] * metrics_non_tubb['recall_0']) / (
            metrics_non_tubb['precision_0'] + metrics_non_tubb['recall_0'] + 1e-8
        )
        avg_f1 = 0.5 * (f1_t + f1_n)

        print(
            f"Epoch {epoch+1}/{args.num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss (Tubb/Non): ({val_loss_tubb:.4f}/{val_loss_non_tubb:.4f}), "
            f"Val F1 (T/N): ({f1_t:.4f}/{f1_n:.4f})"
        )

        # Save best
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                print("Early stopping")
                break

        # TensorBoard metrics
        writer.add_scalar('Loss/Val_Tubb', val_loss_tubb, epoch)
        writer.add_scalar('Accuracy/Val_Tubb', metrics_tubb['accuracy_0'], epoch)
        writer.add_scalar('Precision/Val_Tubb', metrics_tubb['precision_0'], epoch)
        writer.add_scalar('Recall/Val_Tubb', metrics_tubb['recall_0'], epoch)
        writer.add_scalar('Loss/Val_Non_Tubb', val_loss_non_tubb, epoch)
        writer.add_scalar('Accuracy/Val_Non_Tubb', metrics_non_tubb['accuracy_0'], epoch)
        writer.add_scalar('Precision/Val_Non_Tubb', metrics_non_tubb['precision_0'], epoch)
        writer.add_scalar('Recall/Val_Non_Tubb', metrics_non_tubb['recall_0'], epoch)
        writer.flush()

    writer.close()

    # Save history
    metrics_path_tubb = f"metrics_tubb_all_overtones_{args.instrument_overtones}_normalized_{int(args.patch_normalize)}_separate_drums_{int(args.separate_drums)}_targets_{args.num_targets}.json"
    metrics_path_non_tubb = f"metrics_non_tubb_all_overtones_{args.instrument_overtones}_normalized_{int(args.patch_normalize)}_separate_drums_{int(args.separate_drums)}_targets_{args.num_targets}.json"
    json.dump(history, open(metrics_path_tubb, 'w'))
    json.dump(history, open(metrics_path_non_tubb, 'w'))


if __name__ == "__main__":
    main()
