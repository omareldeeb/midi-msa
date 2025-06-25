import torch.nn as nn
import torchvision.models

# Simple CNN boundary classifier
class MobileNetBoundaryClassifier(nn.Module):
    def __init__(self, num_targets=1, pretrained=True):
        super().__init__()

        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        backbone.classifier[-1] = nn.Sequential(
            nn.Linear(backbone.classifier[-1].in_features, num_targets),
        )
        for layer in backbone.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
        self.backbone = backbone
        
    def forward(self, x):
        return self.backbone(x)
