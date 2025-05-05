import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from probes import LinearProbe
from typing import Dict
from torch.utils.data import DataLoader

from project.utils.datasets import tensor_dataset
from project.data.utility import utility_dataframe_interp
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
    df = utility_dataframe_interp(game=game).reset_index(drop=True)
    feature_cols = [f"state_bit{i}" for i in range(64)]
    label_col = "utility"
    df_reduced = df[feature_cols + [label_col]]
    ds = tensor_dataset(df=df_reduced, label="utility")
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
    print(
        f"Output DataFrame has {df_out.shape[0]} rows and "
        f"{D} activation columns" + (f", saved to {into}" if into else "")
    )
    if into:
        df_out.to_csv(into, index=False)

    return df_out


def fit_ols_probes(
    activations_pkl: str,
    output_weights_path: str,
    device: torch.device = None,
    fit_intercept: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load activations from a pickled DataFrame, fit one OLS linear probe per
    interpretable feature, save all probe weights to disk, and return the
    dict of state_dicts.

    Args:
        activations_pkl: path to the pickled DataFrame containing columns
                         ["activation_0",..., "activation_D-1", "interp_..."].
        output_weights_path: file path where to torch.save the probes dict.
        device: torch.device to place the probes on (defaults to CPU).
        fit_intercept: whether to include a bias term in each probe.

    Returns:
        A dict mapping feature names → state_dict of the fitted LinearProbe.
    """
    # 1) set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) load DataFrame
    df = pd.read_pickle(activations_pkl)

    # 3) split activation vs interp columns
    activation_cols = [c for c in df.columns if c.startswith("activation_")]
    interp_cols = [c for c in df.columns if c.startswith("interp_")]

    X = df[activation_cols].values  # shape (N, D)
    N, D = X.shape

    probes_state = {}

    # 4) for each interpretable feature, solve OLS
    for feat in interp_cols:
        y = df[feat].values
        # ensure 2D
        if y.ndim == 1:
            y = y[:, None]  # shape (N, 1)
        # build design matrix
        if fit_intercept:
            X_design = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])
        else:
            X_design = X
        # solve least squares
        w_aug, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        # extract weights & bias
        if fit_intercept:
            W = w_aug[:-1, :].T  # (K, D)
            b = w_aug[-1, :]  # (K,)
        else:
            W = w_aug.T  # (K, D)
            b = np.zeros(y.shape[1], dtype=X.dtype)
        # instantiate probe & assign
        output_dim = y.shape[1]
        probe = LinearProbe(input_dim=D, output_dim=output_dim).to(device)
        probe.linear.weight.data = torch.from_numpy(W).to(device).float()
        probe.linear.bias.data = torch.from_numpy(b).to(device).float()

        probes_state[feat] = probe.state_dict()

    # 5) save all probe weights
    torch.save(probes_state, output_weights_path)
    print(f"Saved {len(probes_state)} OLS probes to {output_weights_path}")

    return probes_state


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
