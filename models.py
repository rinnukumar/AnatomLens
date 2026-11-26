"""Architectures from the original DDP evaluation."""

import torch
from torch import nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 11)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))


# Reconstruct the exact ViT architecture
class SmallViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            1, 128, kernel_size=8, stride=8
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 65, 128)
        )
        self.dropout = nn.Dropout(0.1)

        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=4
        )
        self.norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, 11)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.dropout(x + self.pos_embedding)
        x = self.transformer(x)
        return self.classifier(self.norm(x[:, 0]))


