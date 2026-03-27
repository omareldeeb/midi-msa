# Unified Training and Inference Pipeline

This project uses a single maintained pipeline for both USG and TCN workflows.

## Maintained Entrypoints

- Training: `python midi_msa/train_unified.py ...`
- Inference: `python -m midi_msa.inference ...`

## Configuration

Hydra configs live in:

- `config/config.yaml` (base)
- `config/method/usg.yaml`
- `config/method/tcn.yaml`
- `config/inference.yaml`

Structured config classes are registered in `midi_msa/config/__init__.py`.

## Training Quick Start

### USG

```bash
python midi_msa/train_unified.py \
  method=usg \
  midi_dir=/path/to/lmd_full \
  annotation_dir=/path/to/slms_annotations \
  split_files=[/path/to/cv_split_0.json]
```

### TCN

```bash
python midi_msa/train_unified.py \
  method=tcn \
  midi_dir=/path/to/lmd_full \
  annotation_dir=/path/to/slms_annotations \
  piano_roll_dir=/path/to/piano_roll_cache \
  sslm_dir=/path/to/sslm_cache \
  split_files=[/path/to/cv_split_0.json]
```

### Useful Overrides

```bash
# common
python midi_msa/train_unified.py method=tcn num_epochs=100 lr=5e-4

# device control
python midi_msa/train_unified.py method=usg device=cuda

# wandb logging
python midi_msa/train_unified.py method=tcn wandb.enabled=true wandb.project=midi-msa
```

## Inference Quick Start

```bash
python -m midi_msa.inference \
  method=tcn \
  input_dir=/path/to/midi \
  checkpoint=/path/to/checkpoint.pt \
  output_dir=inference_results
```

The same entrypoint supports `method=usg` with USG-related inference parameters in `config/inference.yaml`.

## Pipeline Components

- `midi_msa/models/registry.py`: selects USG or TCN model from config
- `midi_msa/training/__init__.py`: trainer factory
- `midi_msa/training/base_trainer.py`: shared optimization/checkpoint loop
- `midi_msa/training/usg_trainer.py`: USG-specific dataset/loss logic
- `midi_msa/training/tcn_trainer.py`: TCN-specific dataset/loss logic
- `midi_msa/data/`: active feature extraction and datasets used by unified training/inference

## Cross-Validation

Use one or more split files via `split_files=[...]`.

- Training currently requires `split_files` to be provided.
- When multiple split files are passed, unified training runs k-fold CV.

Example:

```bash
python midi_msa/train_unified.py \
  method=tcn \
  split_files=[/path/to/cv_split_0.json,/path/to/cv_split_1.json]
```

## Troubleshooting

- Ensure dependencies are installed from `requirements.txt`.
- Use absolute paths in Hydra overrides for dataset/cache/checkpoint paths.
- If running from repo root, use `python midi_msa/train_unified.py` (not `python train_unified.py`).
