# Unified Training Pipeline

This document describes the new unified training pipeline for MIDI Music Structure Analysis that supports both USG (patch-based) and TCN (sequence-based) methods.

## Overview

The unified pipeline uses [Hydra](https://hydra.cc/) for configuration management and provides a single entry point for training either method.

### Key Features

- **Single entry point**: One `train_unified.py` script for both methods
- **Hydra configuration**: YAML-based configs with CLI overrides
- **Wandb integration**: Built-in support for experiment tracking
- **Shared components**: Common training loop, checkpointing, and logging
- **Type-safe configs**: Structured configs with validation

## Quick Start

### Install Dependencies

```bash
pip install hydra-core omegaconf wandb torch torchvision pandas tqdm
```

### Train USG Method

```bash
python train_unified.py data_dir=/path/to/data
```

### Train TCN Method

```bash
python train_unified.py \
    method=tcn \
    data_dir=/path/to/data \
    midi_dir=/path/to/midi \
    annotation_dir=/path/to/annotations \
    piano_roll_dir=/path/to/cache
```

## Configuration

### Structure

```
config/
├── config.yaml         # Base configuration
└── method/
    ├── usg.yaml       # USG-specific config
    └── tcn.yaml       # TCN-specific config
```

### Base Parameters

Common parameters shared by both methods:

- `data_dir`: Directory containing data
- `target_ticks_per_beat`: Target resolution (default: 4)
- `instrument_overtones`: Use instrument overtones (default: true)
- `separate_drums`: Separate drum tracks (default: true)
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Number of epochs (default: 50)
- `lr`: Learning rate (default: 1e-3)
- `weight_decay`: Weight decay (default: 1e-4)
- `device`: Device (cuda/cpu/mps, auto-detected if null)

### USG-Specific Parameters

- `window_half_ticks`: Half-window size for patches (default: 256)
- `positive_oversampling_factor`: Oversample positive patches (default: 2)
- `negative_undersampling_factor`: Undersample negative patches (default: 1)
- `patch_normalize`: Normalize patches (default: false)
- `pretrained`: Use pretrained MobileNet (default: false)
- `use_sslm_near`: Use near SSLM features (default: false)
- `use_sslm_far`: Use far SSLM features (default: false)
- `predict_segment_label`: Enable multi-task segment label prediction (default: false)

### TCN-Specific Parameters

- `midi_dir`: Directory with MIDI files (required)
- `annotation_dir`: Directory with annotations (required)
- `piano_roll_dir`: Cache directory for piano rolls (required)
- `sslm_dir`: Cache directory for SSLMs (optional)
- `tcn_layers`: Number of TCN layers (default: 2)
- `compute_beats`: Compute beat activations (default: false)
- `compute_downbeats`: Compute downbeat activations (default: false)
- `compute_segments`: Compute segment boundaries (default: true)
- `beat_loss_weight`: Weight for beat loss (default: 1.0)
- `downbeat_loss_weight`: Weight for downbeat loss (default: 3.0)
- `section_loss_weight`: Weight for section boundary loss (default: 10.0)
- `function_loss_weight`: Weight for segment function loss (default: 1.0)
- `clip_norm`: Gradient clipping norm (default: 1.0)

## CLI Overrides

Hydra allows overriding any config parameter from the command line:

```bash
# Change batch size and learning rate
python train_unified.py batch_size=64 lr=5e-4

# Switch method and override method-specific params
python train_unified.py method=tcn tcn_layers=4 clip_norm=2.0

# Enable wandb logging
python train_unified.py \
    wandb.enabled=true \
    wandb.project=midi-msa \
    wandb.name=my-experiment \
    wandb.tags=[usg,baseline]
```

## Wandb Integration

To enable Weights & Biases logging:

```bash
python train_unified.py \
    wandb.enabled=true \
    wandb.project=your-project \
    wandb.entity=your-entity \
    wandb.name=experiment-name
```

## Cross-Validation

For cross-validation, specify split files:

```bash
python train_unified.py \
    method=tcn \
    split_files=[splits/fold1.json,splits/fold2.json,splits/fold3.json]
```

## Checkpointing

Checkpoints are saved to `checkpoint_dir` (default: `checkpoints/`):

- `checkpoint_epoch_N.pt`: Checkpoint after epoch N
- `best_checkpoint.pt`: Best model based on validation metric

To resume training:

```bash
python train_unified.py resume=true
```

## Custom Configs

Create custom config files for common experiments:

```yaml
# config/my_experiment.yaml
defaults:
  - config
  - method: usg

batch_size: 64
lr: 5e-4
num_epochs: 100

use_sslm_near: true
predict_segment_label: true

wandb:
  enabled: true
  project: midi-msa
  name: usg-with-sslm
```

Then run:

```bash
python train_unified.py --config-name=my_experiment data_dir=/path/to/data
```

## Architecture

### Components

1. **Config System** (`midi_msa/config/`): Hydra-based configuration with structured configs
2. **Base Dataset** (`midi_msa/data/base_dataset.py`): Common MIDI parsing and piano roll creation
3. **Model Registry** (`midi_msa/models/registry.py`): Factory for instantiating models
4. **Base Trainer** (`midi_msa/training/base_trainer.py`): Shared training loop logic
5. **Method Trainers**: USG and TCN specific trainers with data loading and loss computation

### Design Principles

- **Composition over inheritance**: Components are loosely coupled
- **Config-driven**: All hyperparameters in YAML
- **Type-safe**: Structured configs with validation
- **Extensible**: Easy to add new methods or modify existing ones

## Comparison with Original Scripts

| Feature | Original | Unified |
|---------|----------|---------|
| Entry points | `train.py` + `train_tcn.py` | Single `train_unified.py` |
| Configuration | CLI arguments | Hydra YAML + CLI overrides |
| Code sharing | Duplicated logic | Shared base classes |
| Wandb | Manual setup | Built-in integration |
| Extensibility | Modify scripts | Add new method configs |

## Migration Guide

### From old `train.py` (USG)

Old:
```bash
python midi_msa/train.py \
    --data-dir /path/to/data \
    --batch-size 64 \
    --window-half-ticks 256
```

New:
```bash
python train_unified.py \
    data_dir=/path/to/data \
    batch_size=64 \
    window_half_ticks=256
```

### From old `train_tcn.py`

Old:
```bash
python midi_msa/train_tcn.py \
    --midi-dir /path/to/midi \
    --annotation-dir /path/to/annotations \
    --piano-roll-dir /path/to/cache \
    --batch-size 1 \
    --epochs 100
```

New:
```bash
python train_unified.py \
    method=tcn \
    midi_dir=/path/to/midi \
    annotation_dir=/path/to/annotations \
    piano_roll_dir=/path/to/cache \
    batch_size=1 \
    num_epochs=100
```

## Troubleshooting

### Missing Hydra

```bash
pip install hydra-core omegaconf
```

### Config validation errors

Check that required fields are set (marked with `???` in YAML):
- `data_dir` (both methods)
- `midi_dir`, `annotation_dir`, `piano_roll_dir` (TCN method)

### Import errors

Make sure the package is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/midi-msa"
```

Or install in development mode:
```bash
pip install -e .
```

## Next Steps

- Add evaluation scripts following the same unified pattern
- Add hyperparameter sweep support with Hydra's multirun
- Add more method implementations (e.g., Transformer-based)
- Add unit tests for training components
