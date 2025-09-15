from typing import Optional

import torch
import torch.nn as nn
import torchvision.models

# Simple CNN boundary classifier
class MobileNetBoundaryClassifier(nn.Module):
    def __init__(self, num_targets=1, pretrained=True, use_sslm=False):
        super().__init__()

        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        piano_roll_model = torchvision.models.mobilenet_v3_small(weights=weights)
        if use_sslm:
            piano_roll_model.classifier[-1] = nn.Sequential(
                nn.Linear(piano_roll_model.classifier[-1].in_features, 64),
                nn.ReLU(),
            )
        else:
            piano_roll_model.classifier[-1] = nn.Linear(piano_roll_model.classifier[-1].in_features, num_targets)
        
        for layer in piano_roll_model.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
        self.piano_roll_model = piano_roll_model

        if use_sslm:
            sslm_model = torchvision.models.mobilenet_v3_small(weights=weights)
            sslm_model.classifier[-1] = nn.Sequential(
                nn.Linear(sslm_model.classifier[-1].in_features, 64),
                nn.ReLU(),
            )
            for layer in sslm_model.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)
            self.sslm_model = sslm_model
            self.classifier = nn.Sequential(
                nn.Linear(64 * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_targets)
            )

    def forward(self, piano_roll_patch: torch.Tensor, sslm_patch: Optional[torch.Tensor] = None) -> torch.Tensor:
        piano_roll_output = self.piano_roll_model(piano_roll_patch)
        if not hasattr(self, 'sslm_model'):
            return piano_roll_output

        sslm_output = self.sslm_model(torch.cat((sslm_patch, sslm_patch, sslm_patch), dim=1))  # Convert 1-channel to 3-channel
        combined = torch.cat((piano_roll_output, sslm_output), dim=-1)
        return self.classifier(combined)
