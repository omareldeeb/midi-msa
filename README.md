# MIDI-MSA Unified Pipeline

This repository contains the active training and inference pipeline for symbolic music structure analysis using:

- `USG` (patch-based boundary classification)
- `TCN` (sequence-based multi-task modeling)

The current maintained entrypoints are:

- Training: `python midi_msa/train_unified.py ...`
- Inference: `python -m midi_msa.inference ...`

## Project Layout

- `midi_msa/train_unified.py`: unified trainer entrypoint
- `midi_msa/inference.py`: unified inference entrypoint
- `midi_msa/config/`: structured Hydra config registration
- `midi_msa/models/registry.py`: model factory for USG/TCN
- `midi_msa/training/`: trainer factory and method-specific trainers
- `midi_msa/data/`: active dataset and feature utilities
- `config/`: YAML configs (`config.yaml`, `method/usg.yaml`, `method/tcn.yaml`, `inference.yaml`)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

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

## Inference

```bash
python -m midi_msa.inference \
  method=tcn \
  input_dir=/path/to/midi \
  checkpoint=/path/to/best_checkpoint.pt \
  output_dir=inference_results
```

## Notes

- Hydra config overrides are supported for all settings.
- If `wandb` is installed, unified training can log runs when enabled in config.
- For full parameter reference, see `UNIFIED_PIPELINE_README.md` and the files under `config/`.
