#!/usr/bin/env python3
"""
Inference script for MIDI Music Structure Analysis.

Analyzes MIDI files using trained USG or TCN models and outputs
predicted segment boundaries and labels.

Usage:
    # Run inference on a directory of MIDI files
    python -m midi_msa.inference input_dir=/path/to/midi/files checkpoint=/path/to/checkpoint.pt

    # Run with TCN method
    python -m midi_msa.inference method=tcn input_dir=/path/to/midi checkpoint=/path/to/checkpoint.pt

    # Specify output directory and save interval
    python -m midi_msa.inference input_dir=/path/to/midi checkpoint=/path/to/checkpoint.pt output_dir=results save_every=100
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from midi_msa.data.label_preprocessor import LABEL_MAP_TRAIN
from midi_msa.data.utils import (
    compute_sslms,
    create_piano_roll_fast,
    get_piano_roll_cache_path,
    get_sslm_cache_path,
)
from midi_msa.models.registry import build_model


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Method selection
    method: str = "tcn"

    # Input/Output paths
    input_dir: str = ""  # Directory containing MIDI files
    checkpoint: str = ""  # Path to model checkpoint
    output_dir: str = "inference_results"  # Directory to save results

    # Optional cache paths (for faster processing)
    piano_roll_dir: Optional[str] = None
    sslm_dir: Optional[str] = None

    # Processing parameters
    save_every: int = 100  # Save results every N files
    device: str = "auto"  # "auto", "cpu", "cuda", or "mps"

    # Piano roll parameters
    target_ticks_per_beat: int = 4
    instrument_overtones: bool = True
    separate_drums: bool = True

    # SSLM parameters
    use_sslm_near: bool = True
    use_sslm_far: bool = True

    # USG-specific parameters
    window_half_ticks: int = 256
    patch_normalize: bool = False
    num_targets: int = 1

    # USG model parameters
    usg_architecture: str = "usg_original"
    usg_patch_freq_bins: int = 128
    pretrained: bool = False
    output_features: int = 64

    # TCN-specific parameters
    tcn_layers: int = 2
    tcn_kernel_size: int = 5
    conv_filters: int = 20
    function_output_activation: str = "softmax"

    # Prediction parameters
    boundary_threshold: float = 0.5  # Threshold for boundary detection


def register_inference_config():
    """Register inference config with Hydra."""
    cs = ConfigStore.instance()
    cs.store(name="inference_config", node=InferenceConfig)


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def find_midi_files(input_dir: str) -> List[Path]:
    """Find all MIDI files in the input directory recursively."""
    midi_files = []
    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_files.extend(Path(input_dir).rglob(ext))
    return sorted(midi_files)


def process_midi_for_tcn(
    midi_path: Path,
    cfg: DictConfig,
    device: torch.device,
    piano_roll_dir: Optional[Path] = None,
    sslm_dir: Optional[Path] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Process a single MIDI file for TCN inference.

    Returns a dictionary with piano_roll and optional SSLM features.
    """
    file_id = midi_path.stem

    # Check for cached piano roll
    cache_path = None
    if piano_roll_dir:
        cache_path = get_piano_roll_cache_path(
            file_id,
            piano_roll_dir,
            cfg.target_ticks_per_beat,
        )

    if cache_path and cache_path.exists():
        piano_roll = torch.load(cache_path)
    else:
        try:
            piano_roll_np = create_piano_roll_fast(
                path_to_midi_file=str(midi_path),
                chroma=False,
                target_ticks_per_beat=cfg.target_ticks_per_beat,
                instrument_overtones=cfg.instrument_overtones,
                separate_drums=cfg.separate_drums,
            )
            piano_roll = torch.from_numpy(piano_roll_np)

            # Cache if directory is specified
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(piano_roll, cache_path)
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None

    sample = {"piano_roll": piano_roll.unsqueeze(0).to(torch.float32).to(device)}  # Add batch dimension

    # Compute SSLMs if needed
    if cfg.use_sslm_near or cfg.use_sslm_far:
        sslm_cache_path = None
        if sslm_dir:
            sslm_cache_path = get_sslm_cache_path(
                file_id, sslm_dir, cfg.target_ticks_per_beat
            )

        if sslm_cache_path and sslm_cache_path.exists():
            sslm_data = torch.load(sslm_cache_path)
            sslm_near = sslm_data["sslm_near"]
            sslm_far = sslm_data["sslm_far"]
        else:
            # Merge piano roll across channels for SSLM computation
            sslm_piano_roll = piano_roll.sum(dim=0, keepdim=True)
            sslm_near, sslm_far = compute_sslms(
                sslm_piano_roll, L=int((90 / 0.5) * cfg.target_ticks_per_beat)
            )

            # Cache if directory is specified
            if sslm_cache_path:
                sslm_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"sslm_near": sslm_near, "sslm_far": sslm_far}, sslm_cache_path)

        # Match dimensions to piano roll
        num_time_frames = piano_roll.shape[-1]
        height = piano_roll.shape[-2]

        sslm_near = sslm_near[:height, :num_time_frames]
        sslm_far = sslm_far[:height, :num_time_frames]

        # Add channel dimension
        sslm_near = sslm_near.unsqueeze(0)
        sslm_far = sslm_far.unsqueeze(0)

        # Pad height if needed
        if sslm_near.shape[-2] < height:
            pad_amount = height - sslm_near.shape[-2]
            sslm_near = torch.nn.functional.pad(sslm_near, (0, 0, 0, pad_amount))
        if sslm_far.shape[-2] < height:
            pad_amount = height - sslm_far.shape[-2]
            sslm_far = torch.nn.functional.pad(sslm_far, (0, 0, 0, pad_amount))

        if cfg.use_sslm_near:
            sample["sslm_near"] = sslm_near.unsqueeze(0).to(torch.float32).to(device)  # Add batch dimension
        if cfg.use_sslm_far:
            sample["sslm_far"] = sslm_far.unsqueeze(0).to(torch.float32).to(device)  # Add batch dimension

    return sample


def process_midi_for_usg(
    midi_path: Path,
    cfg: DictConfig,
    device: torch.device,
    piano_roll_dir: Optional[Path] = None,
    sslm_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Process a single MIDI file for USG inference.

    Returns piano roll and SSLM data for patch extraction.
    """
    file_id = midi_path.stem

    # Check for cached piano roll
    cache_path = None
    if piano_roll_dir:
        cache_path = get_piano_roll_cache_path(
            file_id,
            piano_roll_dir,
            cfg.target_ticks_per_beat,
        )

    if cache_path and cache_path.exists():
        piano_roll = torch.load(cache_path)
    else:
        try:
            piano_roll_np = create_piano_roll_fast(
                path_to_midi_file=str(midi_path),
                chroma=False,
                target_ticks_per_beat=cfg.target_ticks_per_beat,
                instrument_overtones=cfg.instrument_overtones,
                separate_drums=cfg.separate_drums,
            )
            piano_roll = torch.from_numpy(piano_roll_np)

            # Cache if directory is specified
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(piano_roll, cache_path)
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None

    result = {
        "piano_roll": piano_roll,
        "sslm_near": None,
        "sslm_far": None,
    }

    # Compute SSLMs if needed
    if cfg.use_sslm_near or cfg.use_sslm_far:
        sslm_cache_path = None
        if sslm_dir:
            sslm_cache_path = get_sslm_cache_path(
                file_id, sslm_dir, cfg.target_ticks_per_beat
            )

        if sslm_cache_path and sslm_cache_path.exists():
            sslm_data = torch.load(sslm_cache_path)
            sslm_near = sslm_data["sslm_near"]
            sslm_far = sslm_data["sslm_far"]
        else:
            # Merge piano roll across channels for SSLM computation
            sslm_piano_roll = piano_roll.sum(dim=0, keepdim=True)
            sslm_near, sslm_far = compute_sslms(
                sslm_piano_roll, L=int((90 / 0.5) * cfg.target_ticks_per_beat)
            )

            # Cache if directory is specified
            if sslm_cache_path:
                sslm_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"sslm_near": sslm_near, "sslm_far": sslm_far}, sslm_cache_path)

        result["sslm_near"] = sslm_near
        result["sslm_far"] = sslm_far

    return result


def extract_patch(
    piano_roll: torch.Tensor,
    center_tick: int,
    window_half_ticks: int,
    sslm_near: Optional[torch.Tensor] = None,
    sslm_far: Optional[torch.Tensor] = None,
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:
    """Extract a patch centered at the given tick position."""
    num_ticks = piano_roll.shape[-1]

    # Calculate patch boundaries
    start_tick = max(0, center_tick - window_half_ticks)
    end_tick = min(num_ticks, center_tick + window_half_ticks)

    # Extract and pad patch
    patch = piano_roll[:, :, start_tick:end_tick]

    # Pad if necessary
    target_width = 2 * window_half_ticks
    if patch.shape[-1] < target_width:
        pad_left = max(0, window_half_ticks - center_tick)
        pad_right = target_width - patch.shape[-1] - pad_left
        patch = torch.nn.functional.pad(patch, (pad_left, pad_right))

    if normalize:
        patch = patch / (patch.max() + 1e-8)

    result = {"piano_roll_patch": patch.unsqueeze(0)}  # Add batch dimension

    # Extract SSLM patches if available
    if sslm_near is not None:
        sslm_near_patch = sslm_near[:, start_tick:end_tick]
        if sslm_near_patch.shape[-1] < target_width:
            pad_left = max(0, window_half_ticks - center_tick)
            pad_right = target_width - sslm_near_patch.shape[-1] - pad_left
            sslm_near_patch = torch.nn.functional.pad(sslm_near_patch, (pad_left, pad_right))
        result["sslm_near_patch"] = sslm_near_patch.unsqueeze(0).unsqueeze(0)

    if sslm_far is not None:
        sslm_far_patch = sslm_far[:, start_tick:end_tick]
        if sslm_far_patch.shape[-1] < target_width:
            pad_left = max(0, window_half_ticks - center_tick)
            pad_right = target_width - sslm_far_patch.shape[-1] - pad_left
            sslm_far_patch = torch.nn.functional.pad(sslm_far_patch, (pad_left, pad_right))
        result["sslm_far_patch"] = sslm_far_patch.unsqueeze(0).unsqueeze(0)

    return result


def run_tcn_inference(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    cfg: DictConfig,
    segment_vocab: List[str],
) -> Dict:
    """Run TCN inference on a single sample."""
    model.eval()
    device = next(model.parameters()).device
    
    piano_roll = sample["piano_roll"].to(torch.float32).to(device)
    sslm_near = None
    sslm_far = None
    if "sslm_near" in sample:
        sslm_near = sample["sslm_near"].to(torch.float32).to(device)
    if "sslm_far" in sample:
        sslm_far = sample["sslm_far"].to(torch.float32).to(device)
    with torch.no_grad():
        outputs = model(
            piano_roll,
            sslm_near=sslm_near,
            sslm_far=sslm_far,
        )

    # Get boundary predictions
    boundary_probs = torch.sigmoid(outputs.segment_output).squeeze().cpu().numpy()

    # Get function predictions
    if cfg.function_output_activation == "sigmoid":
        function_probs = torch.sigmoid(outputs.function_outputs).squeeze().cpu().numpy()
    else:
        function_probs = torch.softmax(outputs.function_outputs, dim=1).squeeze().cpu().numpy()

    # Find boundaries above threshold
    boundary_ticks = np.where(boundary_probs > cfg.boundary_threshold)[0]

    # Get predicted labels for each boundary
    predictions = []
    for tick in boundary_ticks:
        label_idx = function_probs[:, tick].argmax()
        label = segment_vocab[label_idx]
        prob = float(boundary_probs[tick])
        predictions.append({
            "tick": int(tick),
            "quarter_note": float(tick) / cfg.target_ticks_per_beat,
            "label": label,
            "boundary_probability": prob,
            "label_probabilities": {
                segment_vocab[i]: float(function_probs[i, tick])
                for i in range(len(segment_vocab))
            },
        })

    return {
        "predictions": predictions,
        "num_ticks": int(boundary_probs.shape[0]),
        "boundary_probabilities": boundary_probs.tolist(),
    }


def run_usg_inference(
    model: torch.nn.Module,
    midi_data: Dict,
    cfg: DictConfig,
    device: torch.device,
    segment_vocab: List[str],
    stride: int = 4,
) -> Dict:
    """
    Run USG inference by sliding a window across the entire piece.

    Args:
        model: Trained USG model
        midi_data: Dictionary with piano_roll and optional SSLM data
        cfg: Configuration
        device: Torch device
        segment_vocab: List of segment labels
        stride: Stride for sliding window (in ticks)
    """
    model.eval()

    piano_roll = midi_data["piano_roll"]
    sslm_near = midi_data["sslm_near"]
    sslm_far = midi_data["sslm_far"]

    num_ticks = piano_roll.shape[-1]
    boundary_probs = np.zeros(num_ticks)
    counts = np.zeros(num_ticks)

    # Slide window across the piece
    positions = list(range(0, num_ticks, stride))

    with torch.no_grad():
        for center_tick in positions:
            patch_data = extract_patch(
                piano_roll,
                center_tick,
                cfg.window_half_ticks,
                sslm_near=sslm_near if cfg.use_sslm_near else None,
                sslm_far=sslm_far if cfg.use_sslm_far else None,
                normalize=cfg.patch_normalize,
            )

            # Move to device
            patch_data = {
                k: v.to(torch.float32).to(device) for k, v in patch_data.items()
            }

            outputs = model(
                patch_data["piano_roll_patch"],
                patch_data.get("sslm_near_patch"),
                patch_data.get("sslm_far_patch"),
            )

            prob = torch.sigmoid(outputs["boundary_logits"]).item()
            boundary_probs[center_tick] += prob
            counts[center_tick] += 1

    # Average overlapping predictions
    mask = counts > 0
    boundary_probs[mask] /= counts[mask]

    # Find boundaries above threshold
    boundary_ticks = np.where(boundary_probs > cfg.boundary_threshold)[0]

    # Build predictions
    predictions = []
    for tick in boundary_ticks:
        predictions.append({
            "tick": int(tick),
            "quarter_note": float(tick) / cfg.target_ticks_per_beat,
            "boundary_probability": float(boundary_probs[tick]),
        })

    return {
        "predictions": predictions,
        "num_ticks": num_ticks,
        "boundary_probabilities": boundary_probs.tolist(),
    }


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    print("=" * 80)
    print("MIDI Music Structure Analysis - Inference")
    print("=" * 80)
    print(f"\nMethod: {cfg.method}")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}\n")

    # Validate inputs
    if not cfg.input_dir:
        raise ValueError("input_dir must be specified")
    if not cfg.checkpoint:
        raise ValueError("checkpoint must be specified")
    if not Path(cfg.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    # Setup device
    device = get_device(cfg.device)
    print(f"Using device: {device}\n")

    # Build segment vocabulary
    model_segment_vocab = sorted(list(set(LABEL_MAP_TRAIN.values())))

    # Build model
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Load checkpoint
    print(f"Loading checkpoint from {cfg.checkpoint}...")
    checkpoint = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Checkpoint loaded successfully.\n")

    # Find MIDI files
    midi_files = find_midi_files(cfg.input_dir)
    print(f"Found {len(midi_files)} MIDI files\n")

    if not midi_files:
        print("No MIDI files found. Exiting.")
        return

    # Setup output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup cache directories
    piano_roll_dir = Path(cfg.piano_roll_dir) if cfg.piano_roll_dir else None
    sslm_dir = Path(cfg.sslm_dir) if cfg.sslm_dir else None

    if piano_roll_dir:
        piano_roll_dir.mkdir(parents=True, exist_ok=True)
    if sslm_dir:
        sslm_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    all_results = {}
    errors = []

    print("Processing MIDI files...")
    for i, midi_path in enumerate(tqdm(midi_files, desc="Inference")):
        try:
            file_id = str(midi_path.relative_to(cfg.input_dir))

            if cfg.method == "tcn":
                sample = process_midi_for_tcn(
                    midi_path, cfg, device, piano_roll_dir, sslm_dir
                )
                if sample is None:
                    errors.append({"file": file_id, "error": "Failed to process MIDI"})
                    continue

                result = run_tcn_inference(model, sample, cfg, model_segment_vocab)
            else:  # USG
                midi_data = process_midi_for_usg(
                    midi_path, cfg, device, piano_roll_dir, sslm_dir
                )
                if midi_data is None:
                    errors.append({"file": file_id, "error": "Failed to process MIDI"})
                    continue

                result = run_usg_inference(
                    model, midi_data, cfg, device, model_segment_vocab
                )

            result["file"] = file_id
            result["method"] = cfg.method
            all_results[file_id] = result

            # Periodic save
            if (i + 1) % cfg.save_every == 0:
                checkpoint_path = output_dir / f"results_checkpoint_{i + 1}.json"
                save_results({"results": all_results, "errors": errors}, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

        except Exception as e:
            errors.append({"file": str(midi_path), "error": str(e)})
            print(f"\nError processing {midi_path}: {e}")

    # Save final results
    final_output = {
        "results": all_results,
        "errors": errors,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "num_processed": len(all_results),
        "num_errors": len(errors),
    }

    final_path = output_dir / "results.json"
    save_results(final_output, final_path)

    print("\n" + "=" * 80)
    print("Inference completed!")
    print(f"Processed: {len(all_results)} files")
    print(f"Errors: {len(errors)} files")
    print(f"Results saved to: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    register_inference_config()
    main()
