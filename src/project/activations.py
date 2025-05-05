import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from torch.utils.data import DataLoader

from project.utils.datasets import tensor_dataset
from project.data.utility import utility_dataframe
from project.utils.checkpoints import models_directory, load_model_epoch
from project.models.mlp import MLPClassifier


def generate_model_activations(game: str, model: str):
    path = f"{models_directory}/{model}/{game}"
    epochs = list_directory(path)
    for e in epochs:
        act = f"{path}/{e}/activations.pkl"
        generate_checkpoint_activations(
            game=game, epoch=e, into=act, layer="relu3"
        )


def generate_checkpoint_activations(
    game: str,
    epoch: str,
    layer: str,
    into: str = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    1) Loads your trained MLPClassifier for (game, epoch).
    2) Registers a forward‐hook on `layer_observed`.
    3) Runs the entire utility_dataframe through the model (no shuffle).
    4) Builds a df with one row per datapoint, original columns + act_<i> cols.
    5) If `into` is set, writes the DataFrame to CSV at that path.
    """

    # --- 1) load model & checkpoint ---
    model = MLPClassifier(input_dim=64, num_classes=3)
    state = load_model_epoch(
        name=f"{MLPClassifier.name()}",
        game=game,
        epoch=epoch,
    )
    model.load_state_dict(state)
    model.eval()

    # --- 2) prepare hook collector ---
    activations = {}

    def get_hook(name):
        def hook(_mod, _inp, output):
            activations[name] = output.detach().cpu().clone()

        return hook

    # attach to exactly the submodule named `layer`
    submods = dict(model.net.named_modules())
    if layer not in submods:
        raise ValueError(f"Layer '{layer}' not found in model.net")
    submods[layer].register_forward_hook(get_hook(layer))

    # --- 3) prepare data & loader ---
    df = utility_dataframe(game=game).reset_index(drop=True)
    ds = tensor_dataset(df=df, label="utility")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # --- 4) run forward & collect in bulk ---
    all_acts = []
    with torch.no_grad():
        for Xb, _ in loader:
            _ = model(Xb)
            # each activations[layer_observed] is (B, D)
            all_acts.append(activations[layer].numpy())

    # concatenate into (N, D)
    acts_arr = np.concatenate(all_acts, axis=0)
    N, D = acts_arr.shape

    # --- 5) build output DataFrame ---
    act_cols = [f"act_{i}" for i in range(D)]
    act_df = pd.DataFrame(acts_arr, columns=act_cols, index=df.index)

    df_out = pd.concat([df, act_df], axis=1)

    # --- 6) optional save ---
    print(df_out)
    if into:
        df_out.to_csv(into, index=False)

    return df_out


# ----------------
# HELPER FUNCTIONS
# ----------------


def list_directory(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return [item.name for item in p.iterdir()]
