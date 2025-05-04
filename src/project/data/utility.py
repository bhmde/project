import pandas as pd

from project.data.sqlite import (
    query_dataloader,
    expand_binary_feature,
    compose,
)


# Produce [64x(state_bit={0,1}), utility={0,1}]
def utility_dataframe(game: str) -> pd.DataFrame:
    return query_dataloader(
        db="solutions.db",
        table=game,
        columns=[
            "utility_0",
            "utility_1",
            "player",
            "state",
        ],
        transform=compose(
            expand_state_vector,
            replace_turn_utility,
            discard_turn,
        ),
    )


# Produce [64x(state_bit={0,1}), utility={0,1}, ...other features]
def utility_dataframe_interp(game: str) -> pd.DataFrame:
    return query_dataloader(
        db="solutions.db",
        table=game,
        transform=compose(
            expand_state_vector,
            replace_turn_utility,
            discard_turn,
        ),
    )


# ----------
# TRANSFORMS
# ----------


def expand_state_vector(df: pd.DataFrame) -> pd.DataFrame:
    return expand_binary_feature(df=df, column="state")


def replace_turn_utility(df: pd.DataFrame) -> pd.DataFrame:
    df["utility"] = df["utility_0"].where(df["player"] == 0, df["utility_1"])
    df["utility"] = df["utility"].where(df["utility"] == 1, 0)
    del df["utility_0"]
    del df["utility_1"]
    return df


def discard_turn(df: pd.DataFrame) -> pd.DataFrame:
    del df["player"]
    return df
