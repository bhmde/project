import sqlite3
import pandas as pd
import numpy as np
from typing import List, Callable, Optional

# Admits a dataframe with feature and label columns. Returns a new dataframe.
TransformFn = Callable[[pd.DataFrame], pd.DataFrame]


def query_dataloader(
    db: str,
    table: str,
    columns: List[str] = None,
    transform: Optional[TransformFn] = None,
) -> pd.DataFrame:

    # 1) Load the requested columns into a df
    query = f"SELECT * FROM {table}"
    if columns is not None:
        query = f"SELECT {', '.join(columns)} FROM {table}"

    with sqlite3.connect(db) as con:
        df = pd.read_sql_query(query, con)

    # 2) Apply optional transform
    if transform is not None:
        df = transform(df)

    return df


# -------------------
# TRANSFORM UTILITIES
# -------------------


def compose(*funcs):
    def _inner(df):
        for fn in funcs:
            df = fn(df)
        return df

    return _inner


def expand_binary_feature(df: pd.DataFrame, column: str) -> pd.DataFrame:

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
    return df2
