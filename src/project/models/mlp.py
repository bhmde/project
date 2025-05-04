import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: list[int] = [128, 64],
        num_classes: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        logits = self.net(x)
        return logits

    def name(self) -> str:
        return "MLPClassifier"
