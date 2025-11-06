from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torchvision.models

# Default vocabulary of segment labels after preprocessing
SEGMENT_LABEL_VOCAB = ["Intro", "Outro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Solo", "Instrumental", "End", "Start"]

# Simple CNN boundary classifier
class MobileNetBoundaryClassifier(nn.Module):
    def __init__(self, num_targets=1, pretrained=True, use_sslm_near=False, use_sslm_far=False, output_features=64, segment_function_vocab: Optional[List[str]] = None):
        super().__init__()

        self.segment_function_vocab = segment_function_vocab
        self.num_segment_classes = len(segment_function_vocab) if segment_function_vocab is not None else 0

        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        piano_roll_model = torchvision.models.mobilenet_v3_small(weights=weights)
        if use_sslm_near or use_sslm_far or segment_function_vocab is not None:
            piano_roll_model.classifier[-1] = nn.Sequential(
                nn.Linear(piano_roll_model.classifier[-1].in_features, output_features),
                nn.ReLU(),
            )
        else:
            piano_roll_model.classifier[-1] = nn.Linear(piano_roll_model.classifier[-1].in_features, num_targets)

        for layer in piano_roll_model.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
        self.piano_roll_model = piano_roll_model

        if use_sslm_near:
            sslm_near_model = torchvision.models.mobilenet_v3_small(weights=weights)
            sslm_near_model.classifier[-1] = nn.Sequential(
                nn.Linear(sslm_near_model.classifier[-1].in_features, output_features),
                nn.ReLU(),
            )
            for layer in sslm_near_model.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)
            self.sslm_near_model = sslm_near_model

        if use_sslm_far:
            sslm_far_model = torchvision.models.mobilenet_v3_small(weights=weights)
            sslm_far_model.classifier[-1] = nn.Sequential(
                nn.Linear(sslm_far_model.classifier[-1].in_features, output_features),
                nn.ReLU(),
            )
            for layer in sslm_far_model.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)
            self.sslm_far_model = sslm_far_model

        output_features_total = output_features + (output_features if use_sslm_near else 0) + (output_features if use_sslm_far else 0)

        # Boundary classification head
        if use_sslm_near or use_sslm_far or segment_function_vocab is not None:
            self.boundary_classifier = nn.Sequential(
                nn.Linear(output_features_total, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_targets)
            )

        # Segment label classification head
        if segment_function_vocab is not None:
            self.segment_label_classifier = nn.Sequential(
                nn.Linear(output_features_total, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_segment_classes)
            )

    def forward(self, piano_roll_patch: torch.Tensor, sslm_near_patch: Optional[torch.Tensor] = None, sslm_far_patch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        piano_roll_output = self.piano_roll_model(piano_roll_patch)

        # If no additional features, return boundary prediction only
        if not hasattr(self, 'sslm_near_model') and not hasattr(self, 'sslm_far_model') and self.segment_function_vocab is None:
            return {"boundary_logits": piano_roll_output}

        # Fuse features from different models
        features = [piano_roll_output]

        if hasattr(self, 'sslm_near_model') and sslm_near_patch is not None:
            sslm_near_output = self.sslm_near_model(torch.cat((sslm_near_patch, sslm_near_patch, sslm_near_patch), dim=1))
            features.append(sslm_near_output)

        if hasattr(self, 'sslm_far_model') and sslm_far_patch is not None:
            sslm_far_output = self.sslm_far_model(torch.cat((sslm_far_patch, sslm_far_patch, sslm_far_patch), dim=1))
            features.append(sslm_far_output)

        combined = torch.cat(features, dim=-1)

        output = {}
        output["boundary_logits"] = self.boundary_classifier(combined)

        if self.segment_function_vocab is not None:
            output["segment_label_logits"] = self.segment_label_classifier(combined)

        return output
