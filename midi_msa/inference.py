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
from typing import Dict, List, Optional, Sequence

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from midi_msa.data.label_preprocessor import LABEL_MAP_TRAIN
from midi_msa.data.utils import (
    compute_sslms_from_midi_path,
    create_piano_roll_fast,
    get_piano_roll_cache_path,
    get_sslm_cache_path,
)
from midi_msa.models.registry import build_model


INFERENCE_ONLY_FIELDS = {
    "input_dir",
    "checkpoint",
    "output_dir",
    "piano_roll_dir",
    "sslm_dir",
    "save_every",
    "device",
    "boundary_threshold",
    "use_checkpoint_config",
}


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Method selection
    method: str = "tcn"

    # Input/Output paths
    input_dir: str = ""
    checkpoint: str = ""
    output_dir: str = "inference_results"

    # Optional cache paths
    piano_roll_dir: Optional[str] = None
    sslm_dir: Optional[str] = None

    # Processing parameters
    save_every: int = 100
    device: str = "auto"
    use_checkpoint_config: bool = True

    # Piano roll parameters
    target_ticks_per_beat: int = 4
    instrument_overtones: bool = True
    separate_drums: bool = True

    # SSLM parameters
    use_sslm_near: bool = True
    use_sslm_far: bool = True

    # Shared model parameters
    dropout_rate: float = 0.1
    compute_segment_labels: bool = True

    # USG-specific parameters
    window_half_ticks: int = 256
    patch_normalize: bool = False
    num_targets: int = 1
    usg_architecture: str = "usg_original"
    usg_patch_freq_bins: int = 128
    pretrained: bool = False
    output_features: int = 64

    # TCN-specific parameters
    tcn_layers: int = 11
    tcn_kernel_size: int = 5
    conv_filters: int = 20
    function_output_activation: str = "softmax"

    # Prediction parameters
    boundary_threshold: float = 0.5


def register_inference_config():
    """Register inference config with Hydra."""
    cs = ConfigStore.instance()
    cs.store(name="inference_config", node=InferenceConfig)


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def find_midi_files(input_dir: str) -> List[Path]:
    """Find all MIDI files in the input directory recursively."""
    midi_files = []
    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_files.extend(Path(input_dir).rglob(ext))
    return sorted(midi_files)


def merge_checkpoint_config(cfg: DictConfig, checkpoint: Dict) -> DictConfig:
    """Use saved training config for model/data settings when available."""
    if not cfg.get("use_checkpoint_config", True):
        return cfg

    checkpoint_cfg = checkpoint.get("config")
    if checkpoint_cfg is None:
        return cfg

    inference_overrides = {
        field: cfg.get(field)
        for field in INFERENCE_ONLY_FIELDS
        if field in cfg
    }
    merged_cfg = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(cfg, resolve=False)),
        OmegaConf.create(checkpoint_cfg),
        OmegaConf.create(inference_overrides),
    )

    if "compute_segment_labels" not in merged_cfg and "compute_segments" in merged_cfg:
        merged_cfg["compute_segment_labels"] = merged_cfg["compute_segments"]

    return merged_cfg


def normalize_piano_roll_dict(data) -> Dict:
    """Normalize cache payloads to the current piano-roll dict format."""
    if isinstance(data, torch.Tensor):
        return {"piano_roll": data, "measure_ticks": None, "time_signatures": []}

    if not isinstance(data, dict) or "piano_roll" not in data:
        raise TypeError("Unsupported piano roll cache format")

    normalized = dict(data)
    if isinstance(normalized["piano_roll"], np.ndarray):
        normalized["piano_roll"] = torch.from_numpy(normalized["piano_roll"])
    return normalized


def consolidate_piano_roll(piano_roll: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Match the piano-roll channel consolidation used by the datasets."""
    if not cfg.separate_drums and cfg.instrument_overtones:
        piano_roll = torch.stack(
            [
                piano_roll[0] + piano_roll[2],
                piano_roll[1],
                torch.zeros_like(piano_roll[0]),
            ]
        )
    elif cfg.separate_drums and not cfg.instrument_overtones:
        piano_roll = torch.stack(
            [
                piano_roll[0],
                torch.zeros_like(piano_roll[0]),
                piano_roll[2],
            ]
        )
    elif not cfg.separate_drums and not cfg.instrument_overtones:
        piano_roll = torch.stack(
            [
                piano_roll[0] + piano_roll[2],
                torch.zeros_like(piano_roll[0]),
                torch.zeros_like(piano_roll[0]),
            ]
        )

    return torch.clip(piano_roll, 0.0, 1.0)


def load_or_compute_piano_roll_dict(
    midi_path: Path,
    cache_file_id: str,
    cfg: DictConfig,
    piano_roll_dir: Optional[Path],
) -> Dict:
    """Load current piano-roll cache or compute it from the MIDI file."""
    cache_path = get_piano_roll_cache_path(
        cache_file_id,
        piano_roll_dir,
        cfg.target_ticks_per_beat,
    )

    if cache_path and cache_path.exists():
        return normalize_piano_roll_dict(torch.load(cache_path, weights_only=False))

    piano_roll_dict = create_piano_roll_fast(
        path_to_midi_file=str(midi_path),
        chroma=False,
        target_ticks_per_beat=cfg.target_ticks_per_beat,
    )
    piano_roll_dict["piano_roll"] = torch.from_numpy(piano_roll_dict["piano_roll"])

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(piano_roll_dict, cache_path)

    return piano_roll_dict


def load_or_compute_sslms(
    midi_path: Path,
    cache_file_id: str,
    cfg: DictConfig,
    sslm_dir: Optional[Path],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load current SSLM cache or compute it using the dataset path."""
    sslm_cache_path = get_sslm_cache_path(
        cache_file_id,
        sslm_dir,
        cfg.target_ticks_per_beat,
    )

    if sslm_cache_path and sslm_cache_path.exists():
        sslm_data = torch.load(sslm_cache_path, weights_only=False)
        return sslm_data["sslm_near"], sslm_data["sslm_far"]

    sslm_near, sslm_far = compute_sslms_from_midi_path(
        p=midi_path,
        target_ticks_per_beat=cfg.target_ticks_per_beat,
    )

    if sslm_cache_path:
        sslm_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"sslm_near": sslm_near, "sslm_far": sslm_far},
            sslm_cache_path,
        )

    return sslm_near, sslm_far


def prepare_sslm_matrix(
    sslm: torch.Tensor,
    height: int,
    num_time_frames: int,
) -> torch.Tensor:
    """Trim and pad SSLMs to match the model input size."""
    sslm = sslm[:height, :num_time_frames]
    if sslm.shape[-2] < height:
        pad_amount = height - sslm.shape[-2]
        sslm = torch.nn.functional.pad(sslm, (0, 0, 0, pad_amount))
    return sslm


def process_midi_for_tcn(
    midi_path: Path,
    cache_file_id: str,
    cfg: DictConfig,
    device: torch.device,
    piano_roll_dir: Optional[Path] = None,
    sslm_dir: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """Process a single MIDI file for TCN inference."""
    piano_roll_dict = load_or_compute_piano_roll_dict(
        midi_path,
        cache_file_id,
        cfg,
        piano_roll_dir,
    )
    piano_roll = consolidate_piano_roll(piano_roll_dict["piano_roll"], cfg)

    sample = {"piano_roll": piano_roll.unsqueeze(0).to(torch.float32).to(device)}

    measure_ticks = piano_roll_dict.get("measure_ticks")
    if measure_ticks is not None:
        sample["measure_ticks"] = torch.tensor(measure_ticks, dtype=torch.long)

    if cfg.use_sslm_near or cfg.use_sslm_far:
        sslm_near, sslm_far = load_or_compute_sslms(
            midi_path,
            cache_file_id,
            cfg,
            sslm_dir,
        )

        height = piano_roll.shape[-2]
        num_time_frames = piano_roll.shape[-1]
        sslm_near = prepare_sslm_matrix(sslm_near, height, num_time_frames)
        sslm_far = prepare_sslm_matrix(sslm_far, height, num_time_frames)

        if cfg.use_sslm_near:
            sample["sslm_near"] = (
                sslm_near.unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
            )
        if cfg.use_sslm_far:
            sample["sslm_far"] = (
                sslm_far.unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
            )

    return sample


def process_midi_for_usg(
    midi_path: Path,
    cache_file_id: str,
    cfg: DictConfig,
    piano_roll_dir: Optional[Path] = None,
    sslm_dir: Optional[Path] = None,
) -> Dict:
    """Process a single MIDI file for USG inference."""
    piano_roll_dict = load_or_compute_piano_roll_dict(
        midi_path,
        cache_file_id,
        cfg,
        piano_roll_dir,
    )
    piano_roll = consolidate_piano_roll(piano_roll_dict["piano_roll"], cfg)

    result = {
        "piano_roll": piano_roll,
        "measure_ticks": piano_roll_dict.get("measure_ticks"),
        "sslm_near": None,
        "sslm_far": None,
    }

    if cfg.use_sslm_near or cfg.use_sslm_far:
        sslm_near, sslm_far = load_or_compute_sslms(
            midi_path,
            cache_file_id,
            cfg,
            sslm_dir,
        )

        height = piano_roll.shape[-2]
        num_time_frames = piano_roll.shape[-1]
        result["sslm_near"] = prepare_sslm_matrix(sslm_near, height, num_time_frames)
        result["sslm_far"] = prepare_sslm_matrix(sslm_far, height, num_time_frames)

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

    start_tick = max(0, center_tick - window_half_ticks)
    end_tick = min(num_ticks, center_tick + window_half_ticks)

    patch = piano_roll[:, :, start_tick:end_tick]
    target_width = 2 * window_half_ticks
    if patch.shape[-1] < target_width:
        pad_left = max(0, window_half_ticks - center_tick)
        pad_right = target_width - patch.shape[-1] - pad_left
        patch = torch.nn.functional.pad(patch, (pad_left, pad_right))

    if normalize:
        patch = patch / (patch.max() + 1e-8)

    result = {"piano_roll_patch": patch.unsqueeze(0)}

    if sslm_near is not None:
        sslm_near_patch = sslm_near[:, start_tick:end_tick]
        if sslm_near_patch.shape[-1] < target_width:
            pad_left = max(0, window_half_ticks - center_tick)
            pad_right = target_width - sslm_near_patch.shape[-1] - pad_left
            sslm_near_patch = torch.nn.functional.pad(
                sslm_near_patch,
                (pad_left, pad_right),
            )
        result["sslm_near_patch"] = sslm_near_patch.unsqueeze(0).unsqueeze(0)

    if sslm_far is not None:
        sslm_far_patch = sslm_far[:, start_tick:end_tick]
        if sslm_far_patch.shape[-1] < target_width:
            pad_left = max(0, window_half_ticks - center_tick)
            pad_right = target_width - sslm_far_patch.shape[-1] - pad_left
            sslm_far_patch = torch.nn.functional.pad(
                sslm_far_patch,
                (pad_left, pad_right),
            )
        result["sslm_far_patch"] = sslm_far_patch.unsqueeze(0).unsqueeze(0)

    return result


def segment_probability_dict(
    label_scores: np.ndarray,
    segment_vocab: Sequence[str],
) -> Dict[str, float]:
    total = float(label_scores.sum())
    if total > 0:
        normalized_scores = label_scores / total
    else:
        normalized_scores = np.ones_like(label_scores) / len(label_scores)
    return {
        segment_vocab[i]: float(normalized_scores[i])
        for i in range(len(segment_vocab))
    }


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
    sslm_near = sample.get("sslm_near")
    sslm_far = sample.get("sslm_far")
    measure_ticks = sample.get("measure_ticks")

    if sslm_near is not None:
        sslm_near = sslm_near.to(torch.float32).to(device)
    if sslm_far is not None:
        sslm_far = sslm_far.to(torch.float32).to(device)

    with torch.no_grad():
        outputs = model(
            piano_roll,
            sslm_near=sslm_near,
            sslm_far=sslm_far,
        )

    boundary_probs = torch.sigmoid(outputs.segment_output).squeeze(0).cpu().numpy()
    if cfg.function_output_activation == "sigmoid":
        function_probs = (
            torch.sigmoid(outputs.function_outputs).squeeze(0).cpu().numpy()
        )
    else:
        function_probs = (
            torch.softmax(outputs.function_outputs, dim=1).squeeze(0).cpu().numpy()
        )

    num_ticks = int(boundary_probs.shape[0])
    if measure_ticks is not None and measure_ticks.numel() > 0:
        if measure_ticks.dim() == 1:
            measure_ticks = measure_ticks.unsqueeze(0)
        boundary_ticks_np, predicted_label_indices = model.compute_predictions(
            output=outputs,
            measure_ticks=measure_ticks,
            threshold=cfg.boundary_threshold,
            function_activation=cfg.function_output_activation,
        )
        boundary_ticks = [int(x) for x in boundary_ticks_np if int(x) < num_ticks]
    else:
        boundary_ticks = [
            int(x) for x in np.where(boundary_probs > cfg.boundary_threshold)[0]
        ]
        predicted_label_indices = np.array(
            [int(function_probs[:, tick].argmax()) for tick in boundary_ticks],
            dtype=int,
        )

    if not boundary_ticks or boundary_ticks[0] != 0:
        boundary_ticks = [0] + boundary_ticks

    predictions = []
    for i, start_tick in enumerate(boundary_ticks):
        end_tick = boundary_ticks[i + 1] if i + 1 < len(boundary_ticks) else num_ticks
        segment_slice = function_probs[:, start_tick:end_tick]
        if segment_slice.shape[-1] == 0:
            segment_slice = function_probs[:, start_tick : start_tick + 1]

        label_scores = segment_slice.mean(axis=-1)
        label_idx = (
            int(predicted_label_indices[i])
            if i < len(predicted_label_indices)
            else int(label_scores.argmax())
        )
        label_probabilities = segment_probability_dict(label_scores, segment_vocab)

        predictions.append(
            {
                "tick": start_tick,
                "quarter_note": float(start_tick) / cfg.target_ticks_per_beat,
                "end_tick": int(end_tick),
                "end_quarter_note": float(end_tick) / cfg.target_ticks_per_beat,
                "label": segment_vocab[label_idx],
                "boundary_probability": float(boundary_probs[start_tick]),
                "label_probability": label_probabilities[segment_vocab[label_idx]],
                "label_probabilities": label_probabilities,
            }
        )

    result = {
        "predictions": predictions,
        "boundary_ticks": boundary_ticks,
        "num_ticks": num_ticks,
        "boundary_probabilities": boundary_probs.tolist(),
    }
    if measure_ticks is not None:
        result["measure_ticks"] = measure_ticks.squeeze(0).tolist()
    return result


def build_usg_predictions(
    records: List[Dict],
    boundary_ticks: List[int],
    segment_vocab: List[str],
    cfg: DictConfig,
    boundary_prob_by_tick: Dict[int, float],
) -> List[Dict]:
    """Aggregate USG boundary-sample predictions into segment predictions."""
    records = sorted(records, key=lambda record: record["center_tick"])
    if not records:
        return []

    predictions = []
    boundary_set = set(boundary_ticks[:-1])
    label_votes = None
    segment_index = 0
    current_start = boundary_ticks[0]

    for record in records:
        tick = record["center_tick"]
        if tick in boundary_set and tick != current_start:
            end_tick = boundary_ticks[segment_index + 1]
            prediction = {
                "tick": int(current_start),
                "quarter_note": float(current_start) / cfg.target_ticks_per_beat,
                "end_tick": int(end_tick),
                "end_quarter_note": float(end_tick) / cfg.target_ticks_per_beat,
                "boundary_probability": float(
                    boundary_prob_by_tick.get(current_start, 0.0)
                ),
            }
            if label_votes is not None:
                label_idx = int(torch.argmax(label_votes).item())
                label_probabilities = segment_probability_dict(
                    label_votes.numpy(),
                    segment_vocab,
                )
                prediction["label"] = segment_vocab[label_idx]
                prediction["label_probability"] = label_probabilities[
                    segment_vocab[label_idx]
                ]
                prediction["label_probabilities"] = label_probabilities
            predictions.append(prediction)
            current_start = tick
            segment_index += 1
            label_votes = None

        if "label_probs" in record:
            if label_votes is None:
                label_votes = torch.zeros_like(record["label_probs"])
            label_votes += record["label_probs"]

    end_tick = boundary_ticks[segment_index + 1]
    final_prediction = {
        "tick": int(current_start),
        "quarter_note": float(current_start) / cfg.target_ticks_per_beat,
        "end_tick": int(end_tick),
        "end_quarter_note": float(end_tick) / cfg.target_ticks_per_beat,
        "boundary_probability": float(boundary_prob_by_tick.get(current_start, 0.0)),
    }
    if label_votes is not None:
        label_idx = int(torch.argmax(label_votes).item())
        label_probabilities = segment_probability_dict(
            label_votes.numpy(),
            segment_vocab,
        )
        final_prediction["label"] = segment_vocab[label_idx]
        final_prediction["label_probability"] = label_probabilities[
            segment_vocab[label_idx]
        ]
        final_prediction["label_probabilities"] = label_probabilities
    predictions.append(final_prediction)

    return predictions


def run_usg_inference(
    model: torch.nn.Module,
    midi_data: Dict,
    cfg: DictConfig,
    device: torch.device,
    segment_vocab: List[str],
    stride: int = 4,
) -> Dict:
    """Run USG inference using the same measure-aligned patch positions as training."""
    del stride
    model.eval()

    piano_roll = midi_data["piano_roll"]
    sslm_near = midi_data["sslm_near"]
    sslm_far = midi_data["sslm_far"]
    measure_ticks = midi_data.get("measure_ticks")

    num_ticks = piano_roll.shape[-1]
    boundary_probs = np.zeros(num_ticks, dtype=np.float32)
    if measure_ticks is not None and len(measure_ticks) > 0:
        positions = sorted(
            {int(tick) for tick in measure_ticks if int(tick) < num_ticks}
        )
    else:
        positions = list(range(0, num_ticks, 4))
    if not positions or positions[0] != 0:
        positions.insert(0, 0)

    records = []
    boundary_prob_by_tick = {}

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
            patch_data = {
                key: value.to(torch.float32).to(device)
                for key, value in patch_data.items()
            }

            outputs = model(
                patch_data["piano_roll_patch"],
                patch_data.get("sslm_near_patch"),
                patch_data.get("sslm_far_patch"),
            )

            boundary_probability = float(
                torch.sigmoid(outputs["boundary_logits"]).squeeze().item()
            )
            boundary_probs[center_tick] = boundary_probability
            boundary_prob_by_tick[center_tick] = boundary_probability

            record = {
                "center_tick": center_tick,
                "boundary_probability": boundary_probability,
            }
            if "segment_label_logits" in outputs:
                record["label_probs"] = (
                    torch.softmax(outputs["segment_label_logits"], dim=-1)
                    .squeeze(0)
                    .cpu()
                )
            records.append(record)

    boundary_ticks = [0]
    for record in records:
        if (
            record["boundary_probability"] >= cfg.boundary_threshold
            and record["center_tick"] not in boundary_ticks
        ):
            boundary_ticks.append(record["center_tick"])
    if num_ticks not in boundary_ticks:
        boundary_ticks.append(num_ticks)
    boundary_ticks = sorted(boundary_ticks)

    predictions = build_usg_predictions(
        records,
        boundary_ticks,
        segment_vocab,
        cfg,
        boundary_prob_by_tick,
    )
    result = {
        "predictions": predictions,
        "boundary_ticks": boundary_ticks,
        "num_ticks": num_ticks,
        "boundary_probabilities": boundary_probs.tolist(),
    }
    if measure_ticks is not None:
        result["measure_ticks"] = [int(tick) for tick in measure_ticks]
    return result


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    if not cfg.input_dir:
        raise ValueError("input_dir must be specified")
    if not cfg.checkpoint:
        raise ValueError("checkpoint must be specified")
    if not Path(cfg.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    device = get_device(cfg.device)
    checkpoint = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    cfg = merge_checkpoint_config(cfg, checkpoint)

    print("=" * 80)
    print("MIDI Music Structure Analysis - Inference")
    print("=" * 80)
    print(f"\nMethod: {cfg.method}")
    print(f"Using device: {device}\n")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}\n")

    model_segment_vocab = sorted(list(set(LABEL_MAP_TRAIN.values())))

    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    print(f"Loading checkpoint from {cfg.checkpoint}...")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Checkpoint loaded successfully.\n")

    midi_files = find_midi_files(cfg.input_dir)
    print(f"Found {len(midi_files)} MIDI files\n")
    if not midi_files:
        print("No MIDI files found. Exiting.")
        return

    input_root = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    piano_roll_dir = Path(cfg.piano_roll_dir) if cfg.piano_roll_dir else None
    sslm_dir = Path(cfg.sslm_dir) if cfg.sslm_dir else None
    if piano_roll_dir:
        piano_roll_dir.mkdir(parents=True, exist_ok=True)
    if sslm_dir:
        sslm_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    errors = []

    print("Processing MIDI files...")
    for i, midi_path in enumerate(tqdm(midi_files, desc="Inference")):
        try:
            relative_path = midi_path.relative_to(input_root)
            file_id = str(relative_path)
            cache_file_id = str(relative_path.with_suffix(""))

            if cfg.method == "tcn":
                sample = process_midi_for_tcn(
                    midi_path,
                    cache_file_id,
                    cfg,
                    device,
                    piano_roll_dir,
                    sslm_dir,
                )
                result = run_tcn_inference(model, sample, cfg, model_segment_vocab)
            else:
                midi_data = process_midi_for_usg(
                    midi_path,
                    cache_file_id,
                    cfg,
                    piano_roll_dir,
                    sslm_dir,
                )
                result = run_usg_inference(
                    model,
                    midi_data,
                    cfg,
                    device,
                    model_segment_vocab,
                )

            result["file"] = file_id
            result["method"] = cfg.method
            all_results[file_id] = result

            if (i + 1) % cfg.save_every == 0:
                checkpoint_path = output_dir / f"results_checkpoint_{i + 1}.json"
                save_results({"results": all_results, "errors": errors}, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

        except Exception as exc:
            errors.append({"file": str(midi_path), "error": str(exc)})
            print(f"\nError processing {midi_path}: {exc}")

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
