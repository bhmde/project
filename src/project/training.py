import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from typing import Optional

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


def fit_ols_probe(
    epoch_dir: str,
    feature: str,
    shuffle: bool,
    device: Optional[torch.device] = None,
    fit_intercept: bool = True,
) -> torch.nn.Module:
    """
    Load activations from `{epoch_dir}/activations.pkl`, fit an OLS probe
    to predict `feature`, and save the probe weights to
    `{epoch_dir}/probes/ols/{feature}.pkl`.

    Args:
        epoch_dir: Directory containing activations.pkl and will be saved.
        feature: The target column name to fit.
        device: torch.device (defaults to CPU or CUDA if available).
        fit_intercept: Whether to include a bias term.

    Returns:
        probe: A LinearProbe with weights set to the OLS solution.
    """

    # 1) determine file paths
    pkl_file = os.path.join(epoch_dir, "activations.pkl")
    weights_file = os.path.join(epoch_dir, "probes", "ols", f"{feature}.pkl")
    os.makedirs(os.path.dirname(weights_file), exist_ok=True)

    # 2) set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) load DataFrame
    df = pd.read_pickle(pkl_file)

    # 4) extract activations and target
    activation_cols = [c for c in df.columns if c.startswith("act")]
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in {pkl_file}")

    X = df[activation_cols].values  # shape (N, D)
    y = df[feature].values  # shape (N,)

    if shuffle:
        np.random.shuffle(y)

    # 5) prepare for OLS
    N, D = X.shape
    y = y[:, None]  # (N,1)
    if fit_intercept:
        X_design = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])
    else:
        X_design = X

    # 6) solve least squares
    w_aug, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    # 7) unpack weights and bias
    if fit_intercept:
        W = w_aug[:-1, :].T  # shape (1, D)
        b = float(w_aug[-1, 0])
    else:
        W = w_aug.T  # shape (1, D)
        b = 0.0

    # 8) create probe and load params
    probe = LinearProbe(input_dim=D, output_dim=1).to(device)
    probe.linear.weight.data = torch.from_numpy(W).to(device).float()
    probe.linear.bias.data = torch.tensor(
        [b], dtype=torch.float32, device=device
    )

    # 9) save state_dict
    torch.save(probe.state_dict(), weights_file)
    print(f"Saved probe for '{feature}' to {weights_file}")

    return probe


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
