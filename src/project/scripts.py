import argparse

from project.training import train_utility_evaluator


# Train a position evaluation MLP on an input game.
def train_evaluator():

    parser = argparse.ArgumentParser(
        prog="train-evaluator",
        description="Train a position evaluation MLP on an input game.",
    )

    parser.add_argument(
        "-g",
        "--game",
        help="Game variant to solve.",
        default="mnk_3_3_3",
    )

    args = parser.parse_args()
    train_utility_evaluator(args.game)
