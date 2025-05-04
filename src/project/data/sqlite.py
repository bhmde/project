import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple, Callable, Optional

# A dataframe, optionally accompanied by a specified label column.
TransformResult = Tuple[pd.DataFrame, Optional[str]]

# Admits a dataframe with feature and label columns. Returns a new dataframe,
# optionally specifying a new label column.
TransformFn = Callable[[pd.DataFrame], TransformResult]


def query_dataloader(
    db: str,
    table: str,
    columns: List[str],
    batch_size: int = 64,
    num_workers: int = 4,
    label: Optional[str] = None,
    transform: Optional[TransformFn] = None,
) -> Tuple[TensorDataset, DataLoader]:

    # 1) Load the requested columns into a DataFrame
    initial_cols = columns
    query = f"SELECT {', '.join(initial_cols)} FROM {table}"
    with sqlite3.connect(db) as con:
        df = pd.read_sql_query(query, con)

    # 2) Apply optional transform
    if transform is not None:
        df, label = transform(df)
        columns = [c for c in df.columns if c != label]

    if label is None:
        raise Exception("Label required for dataset construction.")

    # 3) Cast feature & label columns to int64
    df_features = df[columns].astype("int64")
    df_label = df[label].astype("int64")

    # 4) Convert to torch tensors
    X = torch.tensor(df_features.values, dtype=torch.int64)
    y = torch.tensor(df_label.values, dtype=torch.int64)

    # 5) Build dataset + loader
    ds = TensorDataset(X, y)
    loader = DataLoader(
        ds,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return ds, loader


def compose_transforms(*funcs):
    def _inner(df):
        label = None
        for fn in funcs:
            df, label = fn(df)
        return (df, label)

    return _inner


# ----------
# TRANSFORMS
# ----------


def expand_binary_feature(
    df: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, Optional[str]]:

    # 1) extract the integer column
    arr = df[column].to_numpy(dtype=np.uint64)

    # 2) unpack bits little‑endian into shape
    bytes_arr = arr.view(np.uint8).reshape(-1, 8)
    bits = np.unpackbits(bytes_arr, axis=1, bitorder="little")

    # 3) make new DataFrame of bit columns
    bit_cols = [f"{column}_bit{i}" for i in range(64)]
    bits_df = pd.DataFrame(bits, columns=bit_cols, index=df.index)

    # 4) drop the original integer column, concat bits
    df2 = pd.concat([df.drop(columns=[column]), bits_df], axis=1)
    return (df2, None)
