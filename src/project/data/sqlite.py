import sqlite3
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple


def query_dataloader(
    table: str,
    label: str,
    features: List[str],
    batch_size: int = 64,
    num_workers: int = 4,
    db_path: str = "solutions.db",
) -> Tuple[TensorDataset, DataLoader]:
    """
    Load data from a SQLite table into a PyTorch DataLoader.

    Args:
        table: name of the table to query.
        label: name of the column to use as the label (default "label").
        features: list of column names to use as features.
        batch_size: batch size for DataLoader.
        num_workers: number of worker processes for DataLoader.
        db_path: path to the SQLite database file.

    Returns:
        ds: TensorDataset of (features, labels).
        loader: DataLoader wrapping that dataset.
    """

    # 1) Load the requested columns into a DataFrame
    cols = features + [label]
    query = f"SELECT {', '.join(cols)} FROM {table}"
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, con)
    con.close()

    # 2) Split out features and labels
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df[label].values, dtype=torch.long)

    # 3) Create dataset and loader
    ds = TensorDataset(X, y)
    loader = DataLoader(
        ds,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return ds, loader
