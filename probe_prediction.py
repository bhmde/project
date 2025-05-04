"""Template for checking what a probe predicts on a frozen model."""

import sys, argparse, joblib, torch
from torch.utils.data import DataLoader
from main import GameStateUtilityMLP, GameStateUtilityDataset
from typing import List


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.device("mps")
else:
    torch.device("cpu")

def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Run a linear probe on a frozen model.")
    parser.add_argument("probe", type=str, help="Path to the *.joblib probe file")
    parser.add_argument("model", type=str, help="Path to the *.pth checkpoint")
    args = parser.parse_args(argv)

    print(f"[probe_runner] Loading probe  … {args.probe}")
    probe = joblib.load(args.probe)

    print(f"[probe_runner] Loading model  … {args.model}")
    ckpt = torch.load(args.model, map_location=DEVICE)
    net = GameStateUtilityMLP(hidden_sizes=[128, 64]).to(DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval();  [p.requires_grad_(False) for p in net.parameters()]

    ds = GameStateUtilityDataset("./data/zero_by_2_1000_4_7_13.csv", precompute=True)
    dl = DataLoader(ds, batch_size=512, shuffle=False)

    print("[probe_runner] Collecting activations …")
    acts = []
    def hook(_m, _i, out): acts.append(out.detach().flatten(1).cpu())
    h = net.feature_layers[1].register_forward_hook(hook)

    with torch.no_grad():
        for X, _ in dl:
            _ = net(X.to(DEVICE))
    h.remove()

    X_probe = torch.cat(acts).numpy()
    y_pred   = probe.predict(X_probe)

    print("first 10 predictions:", y_pred[:10])

if __name__ == "__main__":
    main(sys.argv[1:])