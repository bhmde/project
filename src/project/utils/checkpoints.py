import os
import torch
import torch.nn as nn
from typing import OrderedDict

models_directory = "models"


def save_model_epoch(epoch: int, game: str, name: str, model: nn.Module):
    path = f"{models_directory}/{name}/{game}/{epoch}/checkpoint.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_epoch(
    epoch: int, game: str, name: str
) -> OrderedDict[str, torch.Tensor]:
    state = torch.load(
        f"{models_directory}/{name}/{game}/{epoch}/checkpoint.pt"
    )

    return state
