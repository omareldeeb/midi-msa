import torch.nn as nn
from omegaconf import DictConfig

from .mobilenet_boundary_classifier import MobileNetBoundaryClassifier
from .tcn import TCN
from ..data.label_preprocessor import LABEL_MAP_TRAIN, LABEL_MAP_VAL


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Build model based on configuration.

    Args:
        cfg: Hydra config containing method and model parameters

    Returns:
        Initialized model
    """
    # Build segment vocabulary from label map
    segment_vocab_train = sorted(list(set(LABEL_MAP_TRAIN.values())))

    if cfg.method == "usg":
        model = MobileNetBoundaryClassifier(
            num_targets=cfg.num_targets,
            pretrained=cfg.pretrained,
            use_sslm_near=cfg.use_sslm_near,
            use_sslm_far=cfg.use_sslm_far,
            output_features=cfg.output_features,
            segment_function_vocab=segment_vocab_train,
            compute_segment_labels=cfg.compute_segment_labels,
            dropout_rate=cfg.dropout_rate,
        )
    elif cfg.method == "tcn":
        model = TCN(
            piano_roll_channels=3,  # Piano roll channels
            conv_filters=cfg.conv_filters,
            tcn_layers=cfg.tcn_layers,
            tcn_kernel_size=cfg.tcn_kernel_size,
            segment_function_vocab=segment_vocab_train,
            use_sslm_near=cfg.use_sslm_near,
            use_sslm_far=cfg.use_sslm_far,
            dropout_rate=cfg.dropout_rate,
        )
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    return model
