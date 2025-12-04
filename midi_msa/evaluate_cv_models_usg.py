#!/usr/bin/env python3
"""
Evaluation script for MobileNetBoundaryClassifier models trained with cross-validation.

This script evaluates models using the same patch-level approach as training,
and optionally computes segment-level mir_eval metrics.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# import mir_eval.segment
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from midi_msa.data.piano_roll_dataset import PianoRollDataset
from midi_msa.data.utils import get_piano_roll_patches
from midi_msa.models.mobilenet_boundary_classifier import MobileNetBoundaryClassifier, SEGMENT_LABEL_VOCAB
from midi_msa.evaluation.metrics import compute_metrics


def load_split_data(
    split_file: Path,
    data_dir: Path,
    window_half_ticks: int,
    patch_normalize: bool,
    use_sslm_near: bool,
    use_sslm_far: bool,
    segment_function_vocab: Optional[List[str]]
) -> Tuple[DataLoader, List[str]]:
    """
    Load data for a single CV split using the same logic as training.

    Returns:
        DataLoader for validation set
        List of file IDs in validation set
    """
    # Load split
    with open(split_file, 'r') as f:
        split_data = json.load(f)

    val_files = split_data.get('val', [])
    if len(val_files) == 0:
        val_files = split_data.get('test', [])

    # Create temporary split files in data_dir to match expected structure
    # We'll use the validation files and put them in a val folder
    import pandas as pd

    # Load all piano roll patches
    patch_data = get_piano_roll_patches(
        data_dir=data_dir,
        window_half_ticks=window_half_ticks,
        positive_oversampling_factor=1,  # No oversampling for validation
        negative_undersampling_factor=1,
        pad_boundary_patches=True,
        return_sslm_near=use_sslm_near,
        return_sslm_far=use_sslm_far
    )

    piano_rolls = patch_data.piano_rolls
    metadata_dict = patch_data.patch_metadata
    sslm_near_patches = patch_data.sslm_near_patches
    sslm_far_patches = patch_data.sslm_far_patches

    # Convert to dataframe and filter for validation files
    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index')

    # Filter for files in validation set
    metadata_val = metadata_df[metadata_df['filename'].isin(val_files)]
    metadata_val = metadata_val.reset_index(drop=True)

    if len(metadata_val) == 0:
        raise ValueError(f"No validation samples found for files: {val_files[:5]}...")

    # Create dataset
    dataset_val = PianoRollDataset(
        piano_rolls,
        metadata_val,
        normalize=patch_normalize,
        num_targets=1,
        sslm_near_patches=sslm_near_patches,
        sslm_far_patches=sslm_far_patches,
        segment_function_vocab=segment_function_vocab
    )

    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

    return dataloader_val, val_files


def evaluate_split(
    split_file: Path,
    model_checkpoint: Path,
    data_dir: Path,
    args
) -> Dict[str, float]:
    """
    Evaluate a single CV split using patch-level metrics (matching training validation).
    """
    print(f"\n{'='*80}")
    print(f"Evaluating split: {split_file.stem}")
    print(f"{'='*80}")

    # Load model
    device = torch.device(args.device)

    # Determine model configuration
    segment_vocab = SEGMENT_LABEL_VOCAB if args.predict_segment_label else None

    model = MobileNetBoundaryClassifier(
        num_targets=1,
        pretrained=False,
        use_sslm_near=args.use_sslm_near,
        use_sslm_far=args.use_sslm_far,
        output_features=args.output_features,
        segment_function_vocab=segment_vocab
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print(f"Loaded model from {model_checkpoint}")

    # Load data using same logic as training
    try:
        dataloader_val, val_files = load_split_data(
            split_file,
            data_dir,
            args.window_half_ticks,
            args.patch_normalize,
            args.use_sslm_near,
            args.use_sslm_far,
            segment_vocab
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_files': 0,
            'num_patches': 0
        }

    print(f"Validation files: {len(val_files)}")
    print(f"Validation patches: {len(dataloader_val.dataset)}")

    # Evaluate using patch-level metrics (same as training)
    val_outputs = []
    val_targets = []
    val_loss = 0.0

    boundary_criterion = torch.nn.BCEWithLogitsLoss()
    print(len(dataloader_val))

    with torch.no_grad():
        for batch in tqdm(dataloader_val, desc="Evaluating"):
            batch = {k: v.float().to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            output = model(
                batch["piano_roll_patch"],
                batch.get("sslm_near_patch"),
                batch.get("sslm_far_patch")
            )

            boundary_logits = output["boundary_logits"]
            boundary_loss = boundary_criterion(boundary_logits, batch["targets"].float())

            val_outputs.append(boundary_logits)
            val_targets.append(batch["targets"])
            val_loss += boundary_loss.item()

    val_loss /= len(dataloader_val)

    # Compute patch-level metrics (same as training)
    metrics = compute_metrics(
        torch.cat(val_outputs),
        torch.cat(val_targets)
    )

    # Compute F1 from precision and recall
    precision = metrics['precision_0']
    recall = metrics['recall_0']
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    results = {
        'loss': val_loss,
        'accuracy': metrics['accuracy_0'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_files': len(val_files),
        'num_patches': len(dataloader_val.dataset)
    }

    print(f"\nResults:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {metrics['accuracy_0']:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetBoundaryClassifier models with cross-validation"
    )

    # Data paths
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing the piano roll patches (same as used for training)')
    parser.add_argument('--split-files', type=str, nargs='+', required=True,
                        help='JSON files defining CV splits (one per fold)')
    parser.add_argument('--model-checkpoints', type=str, nargs='+', required=True,
                        help='Model checkpoint files (one per fold, same order as split-files)')
    parser.add_argument('--output-file', type=str, default='cv_evaluation_results.json',
                        help='Output JSON file for results')

    # Model configuration (should match training)
    parser.add_argument('--window-half-ticks', type=int, default=256,
                        help='Half window size in ticks for each patch')
    parser.add_argument('--patch-normalize', action='store_true',
                        help='Normalize patches')
    parser.add_argument('--use-sslm-near', action='store_true',
                        help='Use SSLM near patches')
    parser.add_argument('--use-sslm-far', action='store_true',
                        help='Use SSLM far patches')
    parser.add_argument('--output-features', type=int, default=64,
                        help='Number of output features from MobileNet')
    parser.add_argument('--predict-segment-label', action='store_true',
                        help='Whether model predicts segment labels')

    # Device
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')

    args = parser.parse_args()

    # Validate inputs
    if len(args.split_files) != len(args.model_checkpoints):
        raise ValueError("Number of split files must match number of model checkpoints")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Evaluate each split
    fold_results = []
    for i, (split_file, checkpoint) in enumerate(zip(args.split_files, args.model_checkpoints)):
        split_file = Path(split_file)
        checkpoint = Path(checkpoint)

        if not split_file.exists():
            print(f"Warning: Split file not found: {split_file}")
            continue

        if not checkpoint.exists():
            print(f"Warning: Checkpoint not found: {checkpoint}")
            continue

        try:
            metrics = evaluate_split(
                split_file,
                checkpoint,
                data_dir,
                args
            )
            fold_results.append(metrics)
        except Exception as e:
            print(f"Error evaluating fold {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results across folds
    print("\n" + "=" * 80)
    print("Cross-Validation Results")
    print("=" * 80)

    if len(fold_results) == 0:
        print("No results to aggregate!")
        return

    aggregated_metrics = {}
    for metric_name in fold_results[0].keys():
        if metric_name in ['num_files', 'num_patches']:
            continue
        values = [fold[metric_name] for fold in fold_results]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        aggregated_metrics[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'values': values
        }

    # Print summary
    for metric_name, stats in aggregated_metrics.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Values: {[f'{v:.4f}' for v in stats['values']]}")

    # Save results
    output_data = {
        'n_folds': len(fold_results),
        'aggregated_metrics': aggregated_metrics,
        'fold_results': fold_results,
        'config': vars(args)
    }

    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
