import argparse

from project.training import train_utility_evaluator
from project.models.mlp import MLPClassifier
from project.activations import generate_model_activations
from project.utils.visualization import feature_vis


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


def generate_activations():

    parser = argparse.ArgumentParser(
        prog="gen-activations",
        description=(
            "Generate a dataframe of activations per existing checkpoint.",
        ),
    )

    parser.add_argument(
        "-g",
        "--game",
        help="Game variant to generate activations for.",
        default="mnk_3_3_3",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Model to generate activations for.",
        default=f"{MLPClassifier.name()}",
    )

    args = parser.parse_args()
    generate_model_activations(args.game, args.model)


def feature_visualization():
    parser = argparse.ArgumentParser(
        prog="feature-visualization",
        description=("Visualize features from a trained neural network.",),
    )

    parser.add_argument(
        "-g",
        "--game",
        help="Game variant to generate activations for.",
        default="mnk_3_3_3",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Model to generate activations for.",
        default=f"{MLPClassifier.name()}",
    )

    args = parser.parse_args()
    feature_vis(args)
