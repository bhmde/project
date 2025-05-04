import torch
import pandas as pd
from torch.utils.data import TensorDataset


def tensor_dataset(df: pd.DataFrame, label: str):
    features = [c for c in df.columns if c != label]

    df_features = df[features].astype("int64")
    df_label = df[label].astype("int64")

    X = torch.tensor(df_features.values, dtype=torch.int64)
    y = torch.tensor(df_label.values, dtype=torch.int64)

    return TensorDataset(X, y)
