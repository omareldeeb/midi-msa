import torch.nn as nn
import torch.nn.functional as F


class USGBoundaryClassifier(nn.Module):
    def __init__(self, num_targets=1):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(6, 8))
        self.pool1 = nn.MaxPool2d(kernel_size=(6, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 6))
        self.dense = nn.Linear(64 * 18 * 504, 128)
        self.out = nn.Linear(128, num_targets)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense(x))
        return self.out(x)