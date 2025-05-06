import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from threading import Lock
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from project.data.utility import utility_dataframe
from project.models.mlp import MLPClassifier
from project.models.probes import LinearProbe
from project.utils.datasets import tensor_dataset, data_loaders
from project.utils.checkpoints import save_model_epoch


METRICS_DIR = Path("reports")
_METRICS_LOCK = Lock()


def log(name: str, value: float) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metric_file = METRICS_DIR / f"{name}.npy"

    with _METRICS_LOCK:
        if metric_file.exists():
            arr = np.load(metric_file)
        else:
            arr = np.empty((0,), dtype=float)

        arr = np.append(arr, value)
        np.save(metric_file, arr)


def load_metric(name: str) -> np.ndarray:
    metric_file = METRICS_DIR / f"{name}.npy"
    if not metric_file.exists():
        return np.empty((0,), dtype=float)
    return np.load(metric_file)


def train_utility_evaluator(game: str):

    df = utility_dataframe(game=game)
    ds = tensor_dataset(df=df, label="utility")
    t_loader, v_loader = data_loaders(ds=ds, split=0.8, batch=64)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    model = MLPClassifier(input_dim=64, num_classes=3)
    model.to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_epoch(model, t_loader, optimizer, criterion, dev)
        v_loss, v_acc = evaluate(model, v_loader, criterion, dev)

        log("util-eval-epoch-train-loss", t_loss)
        log("util-eval-epoch-valid-loss", v_loss)
        log("util-eval-epoch-train-accu", t_acc)
        log("util-eval-epoch-valid-accu", v_acc)

        print(
            f"Epoch {epoch:02d} "
            f"Train loss: {t_loss:.4f}, acc: {t_acc:.3f} | "
            f"Val loss: {v_loss:.4f}, acc: {v_acc:.3f}"
        )

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

    # 1) determine file paths
    pkl_file = os.path.join(epoch_dir, "activations.pkl")
    weights_file = os.path.join(
        epoch_dir,
        "probes",
        "ols",
        f"{feature}{'_control' if shuffle else ''}.pkl",
    )

    # Make sure dir exists
    os.makedirs(os.path.dirname(weights_file), exist_ok=True)

    # 2) set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # 3) load DataFrame
    print(f"Loading: {pkl_file}")
    df = pd.read_pickle(pkl_file)

    # 4) extract activations and target
    activation_cols = [c for c in df.columns if c.startswith("act")]
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in {pkl_file}")

    X = df[activation_cols].values  # shape (N, D)
    y = df[feature].values  # shape (N,)

    if shuffle:
        np.random.shuffle(y)

    probe, mse = fit_probe_closed_form(X, y, device, fit_intercept)
    torch.save(probe.state_dict(), weights_file)
    print(f"Saved probe for '{feature}' to {weights_file}")

    return mse


def fit_class_probe(
    epoch_dir: str,
    feature: str,
    shuffle: bool,
    device: Optional[torch.device] = None,
    fit_intercept: bool = True,
    num_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> nn.Module:

    # 1) determine file paths
    pkl_file = os.path.join(epoch_dir, "activations.pkl")
    weights_file = os.path.join(
        epoch_dir,
        "probes",
        "multiclass",
        f"{feature}{'_control' if shuffle else ''}.pt",
    )
    os.makedirs(os.path.dirname(weights_file), exist_ok=True)

    # 2) set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # 3) load DataFrame
    print(f"Loading: {pkl_file}")
    df = pd.read_pickle(pkl_file)

    # 4) extract X and y
    activation_cols = [c for c in df.columns if c.startswith("act")]
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in {pkl_file}")

    X_np = df[activation_cols].values.astype(np.float32)  # shape (N, D)
    y_np = df[feature].values.astype(np.int64)  # shape (N,)

    # label‐shuffle control
    if shuffle:
        np.random.shuffle(y_np)

    # convert to torch tensors & dataset
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # build the probe
    dim_in = X.shape[1]
    num_classes = int(y.max().item()) + 1
    probe = nn.Linear(dim_in, num_classes, bias=fit_intercept).to(device)

    # optimizer & loss
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # training loop
    avg_loss = 0
    probe.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = probe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)

    # save the trained probe
    torch.save(probe.state_dict(), weights_file)
    print(f"Saved linear probe for '{feature}' to {weights_file}")

    return avg_loss


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


def fit_probe_closed_form(
    X: np.ndarray, y: np.ndarray, device, fit_intercept: bool = True
):

    # Ensure y is 2D
    if y.ndim == 1:
        y = y[:, None]  # shape (N, 1)

    N, D = X.shape
    _, K = y.shape  # K target dimensions

    # Build design matrix
    if fit_intercept:
        X_design = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])  # (N, D+1)
    else:
        X_design = X  # (N, D)

    # Solve (Xᵀ X) w = Xᵀ y via least squares
    # w_aug shape (D+1, K) if intercept, else (D, K)
    w_aug, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    y_pred = X_design @ w_aug
    mse = np.mean((y_pred - y) ** 2)

    # Extract weights and bias
    if fit_intercept:
        W = w_aug[:-1, :].T  # (K, D)
        b = w_aug[-1, :]  # (K,)
    else:
        W = w_aug.T  # (K, D)
        b = np.zeros(K, dtype=X.dtype)

    # Create probe and assign parameters
    probe = LinearProbe(input_dim=D, output_dim=K).to(device)
    probe.linear.weight.data = torch.from_numpy(W).to(device, torch.float32)
    probe.linear.bias.data = torch.from_numpy(b).to(device, torch.float32)

    return probe, mse
