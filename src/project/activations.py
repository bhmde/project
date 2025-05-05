import pandas as pd
from pathlib import Path
from typing import List
from collections import OrderedDict

from project.data.utility import utility_dataframe
from project.utils.datasets import tensor_dataset, data_loaders
from project.utils.checkpoints import models_directory, load_model_epoch
from project.models.mlp import MLPClassifier

layer_observed = "relu3"
activations = OrderedDict()


def generate_model_activations(game: str, model: str):
    path = f"{models_directory}/{model}/{game}"
    epochs = list_directory(path)
    for e in epochs:
        act = f"{path}/{e}/activations.pkl"
        generate_checkpoint_activations(game=game, epoch=e, into=act)


def generate_checkpoint_activations(game: str, epoch: str, into: str):
    model = MLPClassifier(input_dim=64, num_classes=3)
    state = load_model_epoch(
        name=f"{MLPClassifier.name()}", game=game, epoch=epoch
    )

    model.load_state_dict(state)
    model.eval()

    for name, module in model.net.named_modules():
        if name == layer_observed:
            module.register_forward_hook(get_hook(name))

    df = utility_dataframe(game=game)
    ds = tensor_dataset(df=df, label="utility")
    _, loader = data_loaders(ds=ds, split=0.1, batch=1)

    records = []
    for X_batch, y_batch in loader:
        activations.clear()
        _ = model(X_batch)

        rec = dict()
        act = activations[layer_observed]
        for i in act.numpy():
            for j, val in enumerate(i):
                rec[f"act_{j}"] = float(val)
            rec | X_batch.numpy()
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    print(df.columns)


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


def get_hook(name):
    def hook(module, _inp, out):
        activations[name] = out.detach().cpu()

    return hook
