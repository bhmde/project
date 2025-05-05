import argparse

from project.training import train_utility_evaluator
from project.models.mlp import MLPClassifier
from project.activations import generate_model_activations
from project.utils.plotting import (
    plot_metric_over_epochs,
    plot_multiple_metrics,
)
from project.activations import train_probes_on_checkpoints


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


def train_probes():

    parser = argparse.ArgumentParser(
        prog="train-probes",
        description=(
            "Generate trained linear probes per interpretable feature per pt.",
        ),
    )

    parser.add_argument(
        "-g",
        "--game",
        help="Game variant to train probes for.",
        default="mnk_3_3_3",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Model to train probes for.",
        default=f"{MLPClassifier.name()}",
    )

    args = parser.parse_args()
    train_probes_on_checkpoints(args.game, args.model)


def feature_visualization():
    plot_metric_over_epochs(
        metric="util-eval-epoch-train-loss",
        filename="train-loss",
        y_label="Training Loss",
    )

    accuracy_metrics_info = [
        {"metric": "util-eval-epoch-train-accu", "label": "Training"},
        {"metric": "util-eval-epoch-valid-accu", "label": "Validation"},
    ]

    plot_multiple_metrics(
        metrics_info=accuracy_metrics_info,
        filename="evauator_accuracy_metrics",
        y_label="Accuracy",
    )

    loss_metrics_info = [
        {"metric": "util-eval-epoch-train-loss", "label": "Training"},
        {"metric": "util-eval-epoch-valid-loss", "label": "Validation"},
    ]

    plot_multiple_metrics(
        metrics_info=loss_metrics_info,
        filename="evauator_loss_metrics",
        y_label="Loss",
    )

    probe_mse_info = [
        {
            "metric": "util-eval-epoch-train-loss",
            "label": "Validation Accuracy",
        },
        {"metric": "probe-mse-ply", "label": "ply"},
        {"metric": "probe-mse-corner_count", "label": "corner_count"},
        {"metric": "probe-mse-center_control", "label": "center_control"},
        {"metric": "probe-mse-edge_count", "label": "edge_count"},
        {"metric": "probe-mse-fork_exists", "label": "fork_exists"},
    ]

    plot_multiple_metrics(
        metrics_info=probe_mse_info,
        filename="probe-vs-model",
        y_label="MSE",
    )
