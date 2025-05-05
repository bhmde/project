import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from project.data.utility import utility_dataframe
from project.models.mlp import MLPClassifier
from project.models.probes import LinearProbe
from project.utils.datasets import tensor_dataset, data_loaders
from project.utils.checkpoints import save_model_epoch


def train_utility_evaluator(game: str):

    df = utility_dataframe(game=game)
    ds = tensor_dataset(df=df, label="utility")
    t_loader, v_loader = data_loaders(ds=ds, split=0.8, batch=64)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim=64, num_classes=3)
    model.to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_epoch(model, t_loader, optimizer, criterion, dev)
        v_loss, v_acc = evaluate(model, v_loader, criterion, dev)

        print(
            f"Epoch {epoch:02d} "
            f"Train loss: {t_loss:.4f}, acc: {t_acc:.3f} | "
            f"Val loss: {v_loss:.4f}, acc: {v_acc:.3f}"
        )

        if epoch % 2 == 0:
            save_model_epoch(
                epoch=epoch, game=game, name=MLPClassifier.name(), model=model
            )


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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        if logits.ndim == 1 or logits.size(1) == 1:
            preds = (logits > 0).long()
        else:
            preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item() * X.size(0)
            if logits.ndim == 1 or logits.size(1) == 1:
                preds = (logits > 0).long()
            else:
                preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return running_loss / total, correct / total
