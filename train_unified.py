#!/usr/bin/env python3
"""
Unified training script for MIDI Music Structure Analysis.

Supports both USG (patch-based) and TCN (sequence-based) methods.
Configuration is managed via Hydra - see config/ directory for examples.

Usage:
    # Train USG method (default)
    python train_unified.py data_dir=/path/to/data

    # Train TCN method
    python train_unified.py method=tcn data_dir=/path/to/data midi_dir=/path/to/midi piano_roll_dir=/path/to/cache

    # Override config parameters
    python train_unified.py method=tcn batch_size=64 num_epochs=100

    # Enable wandb logging
    python train_unified.py wandb.enabled=true wandb.project=my-project
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from midi_msa.config import register_configs
from midi_msa.models.registry import build_model
from midi_msa.training import build_trainer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("MIDI Music Structure Analysis - Unified Training Pipeline")
    print("=" * 80)
    print(f"\nMethod: {cfg.method}")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}\n")

    # Set seed if specified
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    # Setup device
    device = torch.device(cfg.device)
    print(f"Using device: {device}\n")

    # Build model
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Load checkpoint if resuming
    if cfg.resume:
        checkpoint_path = cfg.checkpoint_dir / "best_checkpoint.pt"
        if checkpoint_path.exists():
            print(f"Resuming from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Warning: Resume requested but no checkpoint found at {checkpoint_path}")

    # Build trainer
    print("Initializing trainer...")
    trainer = build_trainer(cfg, model, device)

    # Train
    print("\nStarting training...\n")
    trainer.train()

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    register_configs()
    main()
