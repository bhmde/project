import matplotlib.pyplot as plt
from typing import List


def plot_loss(loss: list[float], title: str, xlabel: str, ylabel: str):
    plt.plot(loss)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_accuracy(accuracy: list[float], title: str, xlabel: str, ylabel: str):
    plt.plot(accuracy)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()




def plot_feature_scores(
    features: List[str],
    scores: List[List[float]],
    title: str = "Feature Scores",
    xlabel: str = "Step",
    ylabel: str = "Score",
):
    if len(features) != len(scores):
        raise ValueError(
            f"Number of features ({len(features)}) must match number of score lists ({len(scores)})"
        )

    fig, ax = plt.subplots()
    for feature_name, score_list in zip(features, scores):
        ax.plot(score_list, label=feature_name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.grid(True)
    plt.show()