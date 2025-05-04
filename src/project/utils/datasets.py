import torch
import pandas as pd
from torch.utils.data import random_split, DataLoader, TensorDataset
from typing import Tuple


# Return a torch dataset from a pandas dataframe
def tensor_dataset(df: pd.DataFrame, label: str) -> TensorDataset:
    features = [c for c in df.columns if c != label]

    df_features = df[features].astype("int64")
    df_label = df[label].astype("int64")

    X = torch.tensor(df_features.values, dtype=torch.int64)
    y = torch.tensor(df_label.values, dtype=torch.int64)

    return TensorDataset(X, y)


# Return testing and validation data loaders
def data_loaders(
    ds: TensorDataset, split: float = 0.8, batch: int = 64
) -> Tuple[DataLoader, DataLoader]:

    train_size = int(split * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    t_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=4
    )

    v_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=4
    )

    return t_loader, v_loader
