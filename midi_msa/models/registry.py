import torch.nn as nn
from omegaconf import DictConfig

from .mobilenet_boundary_classifier import MobileNetBoundaryClassifier, SEGMENT_LABEL_VOCAB
from .tcn import TCN


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Build model based on configuration.

    Args:
        cfg: Hydra config containing method and model parameters

    Returns:
        Initialized model
    """
    if cfg.method == "usg":
        segment_vocab = SEGMENT_LABEL_VOCAB if cfg.predict_segment_label else None

        model = MobileNetBoundaryClassifier(
            num_targets=cfg.num_targets,
            pretrained=cfg.pretrained,
            use_sslm_near=cfg.use_sslm_near,
            use_sslm_far=cfg.use_sslm_far,
            output_features=cfg.output_features,
            segment_function_vocab=segment_vocab,
        )

    elif cfg.method == "tcn":
        # Build segment vocabulary from label map
        from ..data.label_preprocessor import LABEL_MAP

        segment_vocab = sorted(list(set(LABEL_MAP.values())))

        model = TCN(
            input_channels=3,  # Piano roll channels
            conv_filters=cfg.conv_filters,
            tcn_layers=cfg.tcn_layers,
            tcn_kernel_size=cfg.tcn_kernel_size,
            segment_function_vocab=segment_vocab,
        )

    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    return model
