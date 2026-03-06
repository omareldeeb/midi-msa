from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models

from midi_msa.data.label_preprocessor import LABEL_MAP_TRAIN


SEGMENT_LABEL_VOCAB = sorted(list(set(LABEL_MAP_TRAIN.values())))


class MobileNetBoundaryClassifier(nn.Module):
    """Boundary classifier with TCN-like multi-modal early fusion."""

    def __init__(
        self,
        num_targets: int = 1,
        pretrained: bool = True,
        use_sslm_near: bool = False,
        use_sslm_far: bool = False,
        output_features: int = 64,
        segment_function_vocab: Optional[List[str]] = None,
        compute_segment_labels: bool = False,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.segment_function_vocab = segment_function_vocab
        self.compute_segment_labels = compute_segment_labels
        self.num_segment_classes = len(segment_function_vocab) if segment_function_vocab is not None else 0
        self.dropout_rate = dropout_rate
        self.use_sslm_near = use_sslm_near
        self.use_sslm_far = use_sslm_far
        self.output_features = output_features
        self.stem_channels = 16

        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None

        self.piano_roll_stem = self._build_modality_stem(in_channels=3, out_channels=self.stem_channels)
        if self.use_sslm_near:
            self.sslm_near_stem = self._build_modality_stem(in_channels=1, out_channels=self.stem_channels)
        if self.use_sslm_far:
            self.sslm_far_stem = self._build_modality_stem(in_channels=1, out_channels=self.stem_channels)

        total_stem_channels = self.stem_channels * (1 + int(self.use_sslm_near) + int(self.use_sslm_far))
        # TCN-style fusion: concatenate frontends, then project before the shared trunk.
        self.stem_projection = nn.Sequential(
            nn.Conv2d(total_stem_channels, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.Hardswish(),
        )

        self.backbone = self._build_backbone(weights=weights)

        self.boundary_classifier = nn.Sequential(
            nn.Linear(output_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_targets),
        )

        if segment_function_vocab is not None and self.compute_segment_labels:
            self.segment_label_classifier = nn.Sequential(
                nn.Linear(output_features, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, self.num_segment_classes),
            )

    def forward(
        self,
        piano_roll_patch: torch.Tensor,
        sslm_near_patch: Optional[torch.Tensor] = None,
        sslm_far_patch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        piano_roll_stem = self.piano_roll_stem(piano_roll_patch)
        stem_features = [piano_roll_stem]

        if self.use_sslm_near:
            if sslm_near_patch is not None:
                stem_features.append(self.sslm_near_stem(sslm_near_patch))
            else:
                stem_features.append(torch.zeros_like(piano_roll_stem))

        if self.use_sslm_far:
            if sslm_far_patch is not None:
                stem_features.append(self.sslm_far_stem(sslm_far_patch))
            else:
                stem_features.append(torch.zeros_like(piano_roll_stem))

        fused_input = torch.cat(stem_features, dim=1)
        fused_input = self.stem_projection(fused_input)
        fused_features = self.backbone(fused_input)

        output = {"boundary_logits": self.boundary_classifier(fused_features)}

        if self.segment_function_vocab is not None and self.compute_segment_labels:
            output["segment_label_logits"] = self.segment_label_classifier(fused_features)

        return output

    def _build_backbone(
        self,
        weights: Optional[torchvision.models.MobileNet_V3_Small_Weights],
    ) -> nn.Module:
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        self._set_backbone_dropout(model)
        model.classifier[-1] = nn.Sequential(
            nn.Linear(model.classifier[-1].in_features, self.output_features),
            nn.ReLU(),
        )
        return model

    @staticmethod
    def _build_modality_stem(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def _set_backbone_dropout(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
