"""
This script trains one linear probe per label on frozen activations from a
previously‑trained `GameStateUtilityMLP`. It saves each probe with *joblib*
and prints basic metrics.

Usage (example)
---------------
```bash
python linear_probes.py \
    --csv  states.csv          # same dataset used for training the MLP
    --model output/2025‑05‑03_18‑41‑12_training/best_model.pth \
    --layer 1                  # index in model.feature_layers to probe
    --outdir probes/           # where *.joblib and metrics.txt are saved
```

Four labels templated.

*   **value**              – game‑theoretic utility class ( –1 / 0 / +1 )
*   **is_win_next**        – binary: current player has an immediate win
*   **plys_to_end**        – integer: optimal distance‑to‑terminal (regression)
*   **symmetry_class**     – categorical ID of board under D₄ symmetries

"""

from __future__ import annotations
import argparse
import joblib
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from main import GameStateUtilityMLP, GameStateUtilityDataset

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# -----------------------------------------------------------------------------
# Label‑function definitions
# -----------------------------------------------------------------------------

def label_fn_value(batch: Tuple[torch.Tensor, torch.Tensor]):
    """Label for game‑theoretic utility class: -1, 0, +1."""
    _x, y_enc = batch
    return y_enc.numpy()


def label_fn_is_win_next(batch):
    """Toy binary label: 1 if utility == +1 else 0."""
    _x, y_enc = batch
    return (y_enc == 2).long().numpy()

def label_fn_plys_to_end(batch):
    """Label for how many plays until the end of the game."""
    pass

def label_fn_symmetry(batch):
    """Label for the symmetry class of the board."""
    pass

LABEL_SPECS: Dict[str, Tuple[Callable, Callable[[], object], Callable]] = {
    "value": (
        label_fn_value,
        lambda: LogisticRegression(penalty="l1", solver="saga", max_iter=5000),
        accuracy_score,
    )
    # "is_win_next": (
    #     label_fn_is_win_next,
    #     lambda: LogisticRegression(penalty="l1", solver="saga", max_iter=5000),
    #     accuracy_score,
    # ),
    # "plys_to_end": (
    #     label_fn_plys_to_end,
    #     lambda: RidgeClassifier(alpha=1.0),
    #     mean_squared_error,
    # ),
    # "symmetry_class": (
    #     label_fn_symmetry,
    #     lambda: LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000),
    #     accuracy_score,
    # ),
}

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def collect_activations(model: torch.nn.Module, dl: DataLoader, layer_idx: int):
    model.eval()
    acts: List[torch.Tensor] = []
    stored_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def hook(_m, _in, out):
        acts.append(out.detach().flatten(1))

    h = model.feature_layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        for X, y in tqdm(dl, desc="[linear_probes] Forward", ncols=80):
            X = X.to(DEVICE)
            stored_batches.append((X.cpu(), y))
            _ = model(X)
    h.remove()
    return torch.cat(acts).cpu().numpy(), stored_batches


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(description="Train one linear probe per label")
    ap.add_argument("--csv", required=True, help="CSV game‑state dataset")
    ap.add_argument("--model", required=True, help="Path to *.pth checkpoint")
    ap.add_argument("--layer", type=int, default=1, help="Index in feature_layers")
    ap.add_argument("--batch", type=int, default=1024, help="Batch size")
    ap.add_argument("--outdir", default="probes", help="Where to save probes")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Dataset -----------------------------------------------------------
    print("[linear_probes] Loading dataset …")
    ds = GameStateUtilityDataset(args.csv, precompute=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # Model -------------------------------------------------------------
    print("[linear_probes] Loading base model checkpoint …")
    ckpt = torch.load(args.model, map_location=DEVICE)
    base = GameStateUtilityMLP(hidden_sizes=[128, 64]).to(DEVICE)
    base.load_state_dict(ckpt["model_state_dict"])
    for p in base.parameters():
        p.requires_grad = False

    # Activations -------------------------------------------------------
    X_all, batches = collect_activations(base, dl, args.layer)
    print(f"[linear_probes] Collected activations: {X_all.shape}\n")

    # Probes ------------------------------------------------------------
    summary_lines: List[str] = []
    for name, (labelfn, ctor, metric_fn) in LABEL_SPECS.items():
        print(f"[linear_probes] Training probe ‹{name}› …")
        # Build labels
        y_all = np.concatenate([labelfn(b) for b in batches])

        # Train/val split
        strat = y_all if len(np.unique(y_all)) < len(y_all) else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=strat
        )

        probe = ctor()
        probe.fit(X_tr, y_tr)

        # Metric
        if metric_fn is mean_squared_error:
            score = metric_fn(y_val, probe.predict(X_val))
            mstr = f"MSE={score:.3f}"
        else:
            score = metric_fn(y_val, probe.predict(X_val))
            mstr = f"ACC={score*100:.2f}%"

        # Save
        p_path = outdir / f"probe_{name}.joblib"
        joblib.dump(probe, p_path)
        line = f"{name:<15} | {mstr} | {p_path.name}"
        summary_lines.append(line)
        print(f"[linear_probes] {line}\n")

    # Summary file
    summary_path = outdir / "metrics.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"[linear_probes] All done!  Metrics → {summary_path}\n")


if __name__ == "__main__":
    main()
