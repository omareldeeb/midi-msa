from abc import ABC, abstractmethod
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""

    def __init__(self, cfg: DictConfig, model: nn.Module, device: torch.device):
        self.cfg = cfg
        self.model = model
        self.device = device

        self.optimizer = self.build_optimizer()
        self.best_val_metric = float("inf") if self.lower_is_better() else 0.0
        self.epochs_no_improve = 0
        self.current_epoch = 0

        self.wandb_run = None
        if cfg.wandb.enabled:
            self.init_wandb()

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                name=self.cfg.wandb.name,
                tags=self.cfg.wandb.tags,
                config=dict(self.cfg),
            )
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            self.cfg.wandb.enabled = False

    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to wandb."""
        if self.wandb_run:
            log_dict = {
                f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()
            }
            log_dict["epoch"] = self.current_epoch
            self.wandb_run.log(log_dict)

    def save_checkpoint(
        self, checkpoint_path: Path, is_best: bool = False, **extra_state
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
            "config": dict(self.cfg),
            **extra_state,
        }

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = checkpoint_path.parent / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_metric = checkpoint["best_val_metric"]
        self.current_epoch = checkpoint["epoch"]
        return checkpoint

    def should_stop_early(self) -> bool:
        """Check if training should stop due to no improvement."""
        patience = self.cfg.get("early_stopping_patience", 10)
        return self.epochs_no_improve >= patience

    def update_best_metric(self, val_metric: float) -> bool:
        """Update best validation metric and return whether it improved."""
        improved = False
        if self.lower_is_better():
            if val_metric < self.best_val_metric:
                self.best_val_metric = val_metric
                improved = True
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
        else:
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                improved = True
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
        return improved

    @abstractmethod
    def lower_is_better(self) -> bool:
        """Return whether lower metric values are better."""
        pass

    @abstractmethod
    def get_dataloaders(self) -> Tuple:
        """Create and return train and validation dataloaders."""
        pass

    @abstractmethod
    def get_dataloaders_for_fold(self, split_file: str) -> Tuple:
        """Create and return train and validation dataloaders for a specific fold.

        Args:
            split_file: Path to the JSON file containing train/val split for this fold.

        Returns:
            Tuple of (train_loader, val_loaders)
        """
        pass

    @abstractmethod
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch. Returns metrics dict."""
        pass

    @abstractmethod
    def validate_epoch(self, val_loaders) -> Dict[str, float]:
        """Validate for one epoch. Returns metrics dict."""
        pass

    @abstractmethod
    def get_val_metric_for_early_stopping(self, val_metrics: Dict[str, float]) -> float:
        """Extract the primary metric for early stopping."""
        pass

    def train(self):
        """Main training loop."""
        train_loader, val_loaders = self.get_dataloaders()

        for epoch in range(self.current_epoch, self.cfg.num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.cfg.num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train metrics: {train_metrics}")
            self.log_metrics(train_metrics, prefix="train")

            # Validate
            val_metrics = self.validate_epoch(val_loaders)
            print(f"Val metrics: {val_metrics}")
            self.log_metrics(val_metrics, prefix="val")

            # Check improvement
            val_metric = self.get_val_metric_for_early_stopping(val_metrics)
            improved = self.update_best_metric(val_metric)

            # Save checkpoint
            checkpoint_path = (
                Path(self.cfg.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            )
            self.save_checkpoint(checkpoint_path, is_best=improved)

            if improved:
                print(f"Validation metric improved to {val_metric:.4f}")
            else:
                print(
                    f"No improvement for {self.epochs_no_improve} epochs (best: {self.best_val_metric:.4f})"
                )

            # Early stopping
            if self.should_stop_early():
                print("Early stopping triggered")
                break

        print(f"\nTraining completed! Best validation metric: {self.best_val_metric:.4f}")

        if self.wandb_run:
            self.wandb_run.finish()

    def k_fold_cross_validate(self, split_files: List[str]) -> Dict[str, float]:
        """Run k-fold cross-validation.

        Args:
            split_files: List of paths to JSON files, each containing train/val split for one fold.

        Returns:
            Dictionary containing aggregated metrics across all folds.
        """
        k = len(split_files)
        print(f"\n{'=' * 80}")
        print(f"Starting {k}-fold cross-validation")
        print(f"{'=' * 80}\n")

        # Store metrics for each fold
        fold_metrics: List[Dict[str, float]] = []
        initial_model_state = copy.deepcopy(self.model.state_dict())

        for fold_idx, split_file in enumerate(split_files):
            print(f"\n{'=' * 80}")
            print(f"Fold {fold_idx + 1}/{k} - Using split file: {split_file}")
            print(f"{'=' * 80}\n")

            # Reset model and optimizer for each fold
            self.model.load_state_dict(copy.deepcopy(initial_model_state))
            self.optimizer = self.build_optimizer()
            self.best_val_metric = float("inf") if self.lower_is_better() else 0.0
            self.epochs_no_improve = 0
            self.current_epoch = 0

            # Get dataloaders for this fold
            train_loader, val_loaders = self.get_dataloaders_for_fold(split_file)

            # Training loop for this fold
            for epoch in range(self.cfg.num_epochs):
                self.current_epoch = epoch
                print(f"\nFold {fold_idx + 1}/{k} - Epoch {epoch + 1}/{self.cfg.num_epochs}")

                # Train
                train_metrics = self.train_epoch(train_loader)
                print(f"Train metrics: {train_metrics}")
                self.log_metrics(
                    {**train_metrics, "fold": fold_idx}, prefix="train"
                )

                # Validate
                val_metrics = self.validate_epoch(val_loaders)
                print(f"Val metrics: {val_metrics}")
                self.log_metrics(
                    {**val_metrics, "fold": fold_idx}, prefix="val"
                )

                # Check improvement
                val_metric = self.get_val_metric_for_early_stopping(val_metrics)
                improved = self.update_best_metric(val_metric)

                # Save checkpoint for this fold
                checkpoint_dir = Path(self.cfg.checkpoint_dir) / f"fold_{fold_idx}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path, is_best=improved, fold=fold_idx)

                if improved:
                    print(f"Validation metric improved to {val_metric:.4f}")
                else:
                    print(
                        f"No improvement for {self.epochs_no_improve} epochs (best: {self.best_val_metric:.4f})"
                    )

                # Early stopping
                if self.should_stop_early():
                    print("Early stopping triggered")
                    break

            # Store final metrics for this fold
            final_val_metrics = self.validate_epoch(val_loaders)
            final_val_metrics["best_val_metric"] = self.best_val_metric
            fold_metrics.append(final_val_metrics)

            print(f"\nFold {fold_idx + 1}/{k} completed!")
            print(f"Best validation metric: {self.best_val_metric:.4f}")

        # Aggregate metrics across folds
        aggregated_metrics = self._aggregate_fold_metrics(fold_metrics)

        # Log aggregated metrics
        print(f"\n{'=' * 80}")
        print(f"{k}-Fold Cross-Validation Results")
        print(f"{'=' * 80}")
        for metric_name, value in aggregated_metrics.items():
            print(f"{metric_name}: {value:.4f}")

        if self.wandb_run:
            self.log_metrics(aggregated_metrics, prefix="cv_aggregate")
            self.wandb_run.finish()

        return aggregated_metrics

    def _aggregate_fold_metrics(
        self, fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate metrics across folds by computing mean and std."""
        if not fold_metrics:
            return {}

        # Get all metric keys from the first fold
        metric_keys = fold_metrics[0].keys()
        aggregated = {}

        for key in metric_keys:
            values = [fm[key] for fm in fold_metrics if key in fm]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)

        return aggregated
