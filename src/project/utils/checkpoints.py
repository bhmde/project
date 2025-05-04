import os
import torch
import torch.nn as nn
from typing import OrderedDict

models = "models"


def save_model_epoch(epoch: int, game: str, name: str, model: nn.Module):
    path = f"{models}/{name}/{game}/{epoch}.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_epoch(
    epoch: int, game: str, name: str
) -> OrderedDict[str, torch.Tensor]:
    model = torch.load(f"{models}/{name}/{game}/{epoch}.pt")
    model.eval()
    return model
