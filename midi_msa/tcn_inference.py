#!/usr/bin/env python3
"""
Simple TCN inference script for MIDI Music Structure Analysis.

Takes a directory containing MIDI files and a TCN checkpoint, processes each file
through the model, and outputs predicted segment boundaries and labels.

Usage:
    python -m midi_msa.tcn_inference /path/to/midi/files /path/to/checkpoint.pt

    # With output directory
    python -m midi_msa.tcn_inference /path/to/midi/files /path/to/checkpoint.pt --output-dir results

    # With custom threshold
    python -m midi_msa.tcn_inference /path/to/midi/files /path/to/checkpoint.pt --threshold 0.3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from midi_msa.data.label_preprocessor import LABEL_MAP
from midi_msa.data.utils import create_piano_roll_fast, compute_sslms
from midi_msa.models.tcn import TCN, TCNOutput


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_midi_files(input_dir: Path) -> List[Path]:
    """Find all MIDI files in the input directory recursively."""
    midi_files = []
    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_files.extend(input_dir.rglob(ext))
    return sorted(midi_files)


def load_tcn_model(
    checkpoint_path: Path,
    device: torch.device,
    segment_vocab: List[str],
    use_sslm_near: bool = False,
    use_sslm_far: bool = False,
) -> TCN:
    """Load a TCN model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to extract model config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        model = TCN(
            segment_function_vocab=segment_vocab,
            use_sslm_near=config.get("use_sslm_near", use_sslm_near),
            use_sslm_far=config.get("use_sslm_far", use_sslm_far),
            tcn_layers=config.get("tcn_layers", 2),
            tcn_kernel_size=config.get("tcn_kernel_size", 5),
            conv_filters=config.get("conv_filters", 20),
        )
    else:
        # Use defaults
        model = TCN(
            segment_function_vocab=segment_vocab,
            use_sslm_near=use_sslm_near,
            use_sslm_far=use_sslm_far,
        )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def process_midi_file(
    midi_path: Path,
    model: TCN,
    device: torch.device,
    segment_vocab: List[str],
    target_ticks_per_beat: int = 4,
    threshold: float = 0.5,
) -> Optional[Dict]:
    """
    Process a single MIDI file through the TCN model.

    Returns a dictionary with predictions or None if processing failed.
    """
    try:
        # Create piano roll from MIDI
        result = create_piano_roll_fast(
            path_to_midi_file=str(midi_path),
            chroma=False,
            target_ticks_per_beat=target_ticks_per_beat,
            compute_measure_ticks=True,
        )

        piano_roll = result["piano_roll"]
        measure_ticks = result["measure_ticks"]

        # Convert to tensors
        piano_roll_tensor = torch.from_numpy(piano_roll).float().unsqueeze(0).to(device)
        measure_ticks_tensor = torch.tensor(measure_ticks).unsqueeze(0)

        # Compute SSLMs if model uses them
        sslm_near = None
        sslm_far = None
        if model.use_sslm_near or model.use_sslm_far:
            # Merge piano roll across channels for SSLM computation
            sslm_piano_roll = torch.from_numpy(piano_roll).sum(dim=0, keepdim=True)
            sslm_near_np, sslm_far_np = compute_sslms(
                sslm_piano_roll, L=int((90 / 0.5) * target_ticks_per_beat)
            )

            # Match dimensions to piano roll
            num_time_frames = piano_roll.shape[-1]
            height = piano_roll.shape[-2]

            sslm_near_np = sslm_near_np[:height, :num_time_frames]
            sslm_far_np = sslm_far_np[:height, :num_time_frames]

            # Pad height if needed
            if sslm_near_np.shape[0] < height:
                pad_amount = height - sslm_near_np.shape[0]
                sslm_near_np = np.pad(sslm_near_np, ((0, pad_amount), (0, 0)))
            if sslm_far_np.shape[0] < height:
                pad_amount = height - sslm_far_np.shape[0]
                sslm_far_np = np.pad(sslm_far_np, ((0, pad_amount), (0, 0)))

            if model.use_sslm_near:
                sslm_near = torch.from_numpy(sslm_near_np).float().unsqueeze(0).unsqueeze(0).to(device)
            if model.use_sslm_far:
                sslm_far = torch.from_numpy(sslm_far_np).float().unsqueeze(0).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output: TCNOutput = model(piano_roll_tensor, sslm_near=sslm_near, sslm_far=sslm_far)

        # Decode predictions using model's compute_predictions method
        boundary_ticks, label_indices = model.compute_predictions(
            output, measure_ticks_tensor, threshold=threshold
        )

        # Convert to labels
        predictions = []
        for i, (tick, label_idx) in enumerate(zip(boundary_ticks, label_indices)):
            predictions.append({
                "tick": int(tick),
                "quarter_note": float(tick) / target_ticks_per_beat,
                "label": segment_vocab[label_idx],
            })

        return {
            "file": midi_path.name,
            "predictions": predictions,
            "num_ticks": int(piano_roll.shape[-1]),
            "num_measures": len(measure_ticks),
        }

    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run TCN inference on MIDI files for music structure analysis."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing MIDI files",
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to TCN model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: print to stdout)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Boundary detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--target-ticks-per-beat",
        type=int,
        default=4,
        help="Target ticks per beat for piano roll (default: 4)",
    )
    parser.add_argument(
        "--use-sslm",
        action="store_true",
        help="Use SSLM features (requires model trained with SSLMs)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    # Build segment vocabulary
    segment_vocab = sorted(list(set(LABEL_MAP.values())))
    print(f"Segment vocabulary: {segment_vocab}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_tcn_model(
        args.checkpoint,
        device,
        segment_vocab,
        use_sslm_near=args.use_sslm,
        use_sslm_far=args.use_sslm,
    )
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Find MIDI files
    midi_files = find_midi_files(args.input_dir)
    print(f"Found {len(midi_files)} MIDI files")

    if not midi_files:
        print("No MIDI files found. Exiting.")
        return

    # Process files
    all_results = {}
    errors = []

    for midi_path in tqdm(midi_files, desc="Processing"):
        result = process_midi_file(
            midi_path,
            model,
            device,
            segment_vocab,
            target_ticks_per_beat=args.target_ticks_per_beat,
            threshold=args.threshold,
        )

        if result is not None:
            file_key = str(midi_path.relative_to(args.input_dir))
            all_results[file_key] = result

            # Print predictions for each file
            if args.output_dir is None:
                print(f"\n{midi_path.name}:")
                for pred in result["predictions"]:
                    print(f"  {pred['quarter_note']:.2f} QN (tick {pred['tick']}): {pred['label']}")
        else:
            errors.append(str(midi_path))

    # Save results if output directory specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "results.json"

        final_output = {
            "results": all_results,
            "errors": errors,
            "config": {
                "checkpoint": str(args.checkpoint),
                "threshold": args.threshold,
                "target_ticks_per_beat": args.target_ticks_per_beat,
            },
            "num_processed": len(all_results),
            "num_errors": len(errors),
        }

        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\nProcessed: {len(all_results)} files")
    if errors:
        print(f"Errors: {len(errors)} files")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


if __name__ == "__main__":
    main()
