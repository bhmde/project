import torch
import torch.nn as nn
from collections import OrderedDict

class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        num_classes: int = 3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(input_dim, 128)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("fc2", nn.Linear(128, 128)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("fc3", nn.Linear(128, 64)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("head", nn.Linear(64, num_classes)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        logits = self.net(x)
        return logits

    def name() -> str:
        return "MLPClassifier"
