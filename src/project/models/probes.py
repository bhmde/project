import numpy as np
import torch
from probes import LinearProbe


def fit_probe_closed_form(
    X: np.ndarray, y: np.ndarray, device, fit_intercept: bool = True
):
    """
    Fit a LinearProbe in closed form via ordinary least squares.

    Args:
        X: array of shape (N, D) — activation features
        y: array of shape (N,) or (N, K) — target values
        device: torch device for the probe
        fit_intercept: whether to include a bias term

    Returns:
        probe: LinearProbe with weights and bias set to the OLS solution
    """
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

    return probe
