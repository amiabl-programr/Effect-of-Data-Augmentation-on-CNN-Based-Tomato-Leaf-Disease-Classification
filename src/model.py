"""
CNN classifier .

It's a fixed, consistent platform for
isolating the effect of augmentation. Outputs raw logits; softmax is applied
implicitly by nn.CrossEntropyLoss during training and explicitly during
inference/evaluation.
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes), 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
