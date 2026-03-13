from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class USGBoundaryClassifier(nn.Module):
    """
    Ulrich/Schlueter/Grill CNN (model D from Grill & Schlueter, ISMIR 2015).

    The architecture applies an independent first conv/pool block per input feature,
    concatenates these features frequency-wise, performs frequency maxout over
    non-overlapping bins of 4, and then applies a late second conv layer.
    """

    def __init__(
        self,
        num_targets: int = 1,
        use_sslm_near: bool = False,
        use_sslm_far: bool = False,
        patch_freq_bins: int = 128,
        patch_time_steps: int = 512,
        segment_function_vocab: Optional[List[str]] = None,
        compute_segment_labels: bool = False,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.use_sslm_near = use_sslm_near
        self.use_sslm_far = use_sslm_far
        self.segment_function_vocab = segment_function_vocab
        self.compute_segment_labels = compute_segment_labels
        self.dropout_rate = dropout_rate
        self.num_segment_classes = (
            len(segment_function_vocab) if segment_function_vocab is not None else 0
        )

        # First feature processing layer (independent per input feature).
        self.piano_roll_stem = self._build_first_layer(in_channels=3)
        if self.use_sslm_near:
            self.sslm_near_stem = self._build_first_layer(in_channels=1)
        if self.use_sslm_far:
            self.sslm_far_stem = self._build_first_layer(in_channels=1)

        # Late processing layer after frequency-wise concatenation + maxout.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 6))

        dense_in_features = self._compute_dense_in_features(
            patch_freq_bins=patch_freq_bins,
            patch_time_steps=patch_time_steps,
        )
        self.shared_head = nn.Sequential(
            nn.Linear(dense_in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.boundary_classifier = nn.Linear(128, num_targets)

        if segment_function_vocab is not None and self.compute_segment_labels:
            self.segment_label_classifier = nn.Linear(128, self.num_segment_classes)

    def forward(
        self,
        piano_roll_patch: torch.Tensor,
        sslm_near_patch: Optional[torch.Tensor] = None,
        sslm_far_patch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        stem_features = [self.piano_roll_stem(piano_roll_patch)]

        if self.use_sslm_near:
            if sslm_near_patch is not None:
                stem_features.append(self.sslm_near_stem(sslm_near_patch))
            else:
                stem_features.append(torch.zeros_like(stem_features[0]))

        if self.use_sslm_far:
            if sslm_far_patch is not None:
                stem_features.append(self.sslm_far_stem(sslm_far_patch))
            else:
                stem_features.append(torch.zeros_like(stem_features[0]))

        # Model D fusion: frequency-wise concat followed by maxout over 4 bins.
        x = torch.cat(stem_features, dim=2)
        x = F.max_pool2d(x, kernel_size=(4, 1), stride=(4, 1))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        shared_features = self.shared_head(x)

        output = {"boundary_logits": self.boundary_classifier(shared_features)}
        if self.segment_function_vocab is not None and self.compute_segment_labels:
            output["segment_label_logits"] = self.segment_label_classifier(
                shared_features
            )
        return output

    @staticmethod
    def _build_first_layer(in_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(6, 8)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(6, 3)),
        )

    def _compute_dense_in_features(
        self,
        patch_freq_bins: int,
        patch_time_steps: int,
    ) -> int:
        num_modalities = 1 + int(self.use_sslm_near) + int(self.use_sslm_far)

        # Stem output size after Conv2d(k=(6,8), s=1) and MaxPool2d(k=(6,3), s=(6,3)).
        stem_freq = ((patch_freq_bins - 6) // 1 + 1 - 6) // 6 + 1
        stem_time = ((patch_time_steps - 8) // 1 + 1 - 3) // 3 + 1

        fused_freq = stem_freq * num_modalities
        maxout_freq = (fused_freq - 4) // 4 + 1

        # Late Conv2d(k=(3,6), s=1).
        conv2_freq = maxout_freq - 3 + 1
        conv2_time = stem_time - 6 + 1

        if conv2_freq <= 0 or conv2_time <= 0:
            raise ValueError(
                "Invalid patch dimensions for USGBoundaryClassifier: "
                f"patch_freq_bins={patch_freq_bins}, patch_time_steps={patch_time_steps}, "
                f"use_sslm_near={self.use_sslm_near}, use_sslm_far={self.use_sslm_far}"
            )

        return 64 * conv2_freq * conv2_time
