from dataclasses import dataclass, field
from typing import Optional, List
from hydra.core.config_store import ConfigStore


@dataclass
class WandBConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class BaseConfig:
    method: str = "usg"

    # Data paths
    data_dir: Optional[str] = None
    annotation_dir: Optional[str] = None
    midi_dir: Optional[str] = None

    # Piano roll parameters
    target_ticks_per_beat: int = 4
    instrument_overtones: bool = True
    separate_drums: bool = True

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Device and paths
    device: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    # Cross-validation
    split_files: Optional[List[str]] = None
    val_split: float = 0.1

    # Logging
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # System
    num_workers: int = 4
    seed: Optional[int] = None
    resume: bool = False


@dataclass
class USGConfig(BaseConfig):
    method: str = "usg"

    # USG-specific data parameters
    window_half_ticks: int = 256
    positive_oversampling_factor: int = 2
    negative_undersampling_factor: int = 1
    pad_boundary_patches: bool = True
    patch_normalize: bool = False

    # USG-specific model parameters
    pretrained: bool = False
    output_features: int = 64
    num_targets: int = 1

    # SSLM features
    use_sslm_near: bool = False
    use_sslm_far: bool = False

    # Multi-task learning
    predict_segment_label: bool = False
    segment_label_loss_weight: float = 1.0

    # Data format parameters (for auto-creation)
    split_file: Optional[str] = None
    markers_qn_path: Optional[str] = None
    measures_qn_path: Optional[str] = None
    compute_sslm_near_dataset: bool = False
    compute_sslm_far_dataset: bool = False


@dataclass
class TCNConfig(BaseConfig):
    method: str = "tcn"

    # TCN-specific paths
    piano_roll_dir: str = "???"
    sslm_dir: Optional[str] = None

    # TCN model parameters
    tcn_layers: int = 2
    tcn_kernel_size: int = 5
    conv_filters: int = 20

    # Task configuration
    compute_beats: bool = False
    compute_downbeats: bool = False
    compute_segments: bool = True

    # Loss weights
    beat_loss_weight: float = 1.0
    downbeat_loss_weight: float = 3.0
    section_loss_weight: float = 10.0
    function_loss_weight: float = 1.0

    # Training
    clip_norm: float = 1.0
    save_every: int = 10

    # Transpose augmentation
    transpose_augmentation: bool = True


def register_configs():
    """Register structured configs with Hydra."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)
    cs.store(group="method", name="usg", node=USGConfig)
    cs.store(group="method", name="tcn", node=TCNConfig)
