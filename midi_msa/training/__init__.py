from .usg_trainer import USGTrainer
from .tcn_trainer import TCNTrainer


def build_trainer(cfg, model, device):
    """Factory function to create trainer based on method."""
    if cfg.method == "usg":
        return USGTrainer(cfg, model, device)
    elif cfg.method == "tcn":
        return TCNTrainer(cfg, model, device)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")
