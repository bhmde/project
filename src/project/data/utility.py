import pandas as pd
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader

from data.sqlite import (
    query_dataloader,
    expand_binary_feature,
    compose_transforms,
    TransformResult,
)


def utility_dataset(
    game: str, batch_size: int
) -> Tuple[TensorDataset, DataLoader]:
    ds, loader = query_dataloader(
        db="solutions.db",
        table=game,
        columns=[
            "utility_p0",
            "utility_p1",
            "state",
            "turn",
        ],
        batch_size=batch_size,
        transform=compose_transforms(
            expand_state_vector,
            select_turn_utility,
        ),
    )


# ----------
# TRANSFORMS
# ----------


def expand_state_vector(df: pd.DataFrame) -> TransformResult:
    return expand_binary_feature(df=df, column="state")


def select_turn_utility(df: pd.DataFrame) -> TransformResult:
    df["utility"] = df["utility_p0"].where(df["turn"] == 0, df["utility_p1"])
    return df, "utility"
