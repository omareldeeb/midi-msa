from typing import Optional

import torch
import torch.nn as nn
import torchvision.models

# Simple CNN boundary classifier
class MobileNetBoundaryClassifier(nn.Module):
    def __init__(self, num_targets=1, pretrained=True, use_sslm_near=False, use_sslm_far=False, output_features=64):
        super().__init__()

        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        piano_roll_model = torchvision.models.mobilenet_v3_small(weights=weights)
        if use_sslm_near or use_sslm_far:
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

        self.classifier = nn.Sequential(
            nn.Linear(output_features_total, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_targets)
        )

    def forward(self, piano_roll_patch: torch.Tensor, sslm_patch: Optional[torch.Tensor] = None) -> torch.Tensor:
        piano_roll_output = self.piano_roll_model(piano_roll_patch)
        if not hasattr(self, 'sslm_near_model') and not hasattr(self, 'sslm_far_model'):
            return piano_roll_output

        sslm_near_output = self.sslm_near_model(torch.cat((sslm_patch, sslm_patch, sslm_patch), dim=1))  # Convert 1-channel to 3-channel
        sslm_far_output = self.sslm_far_model(torch.cat((sslm_patch, sslm_patch, sslm_patch), dim=1))  # Convert 1-channel to 3-channel
        combined = torch.cat((piano_roll_output, sslm_near_output, sslm_far_output), dim=-1)
        return self.classifier(combined)
