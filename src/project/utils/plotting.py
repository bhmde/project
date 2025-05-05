import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict

from project.training import load_metric


directory = "reports/visualization"


def plot_metric_over_epochs(
    metric: str, filename: str, y_label: Optional[str] = None
) -> None:
    # 1) Load data
    values = load_metric(metric)
    if values.size == 0:
        raise ValueError(f"No data found for metric '{metric}'")
    epochs = np.arange(1, len(values) + 1)

    # 2) Prepare output path
    out_file = Path(directory) / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 3) Apply a clean, scientific style
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "font.size": 12,
            "axes.linewidth": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "text.usetex": False,
            "lines.linewidth": 1.5,
            "lines.markeredgewidth": 0.5,
        }
    )

    # 4) Plot
    fig, ax = plt.subplots()
    ax.plot(epochs, values, marker="o", markersize=5)
    ax.set_xlabel("Epoch", labelpad=8)
    # Use custom y_label or default
    y_lbl = (
        y_label
        if y_label is not None
        else metric.replace("_", " ").capitalize()
    )
    ax.set_ylabel(y_lbl, labelpad=8)
    ax.set_xlim(1, epochs[-1])
    ax.margins(x=0)

    # Remove top and right spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Light grid on y‑axis only
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)

    # 5) Save
    fig.tight_layout(pad=0.1)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot for '{metric}' to {out_file}.")


def plot_multiple_metrics(
    metrics_info: List[Dict[str, str]],
    filename: str,
    y_label: Optional[str] = None,
) -> None:
    """
    Plots multiple metrics over epochs in a scientific style, with each metric
    plotted in a different color and annotated via a legend beneath the graph.

    Args:
        metrics_info: List of dicts {"metric": metric_name, "label": legend_label}
        filename:     Path and filename for the plot (e.g., 'plots/metrics.png').
        y_label:      Custom Y-axis label. Defaults to 'Value'.
    """
    # Prepare output path
    out_file = Path(directory) / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure style
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "font.family": "serif",
            "font.size": 10,  # smaller base font
            "axes.linewidth": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "text.usetex": False,
            "lines.linewidth": 1,
            "legend.fontsize": 8,
        }
    )

    # Plot each metric
    fig, ax = plt.subplots()
    for info in metrics_info:
        name = info["metric"]
        label = info["label"]
        values = load_metric(name)
        if values.size == 0:
            raise ValueError(f"No data found for metric '{name}'")
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, label=label)

    # Labels and limits
    ax.set_xlabel("Epoch", labelpad=6)
    y_lbl = y_label if y_label is not None else "Value"
    ax.set_ylabel(y_lbl, labelpad=6)
    max_epoch = max(len(load_metric(info["metric"])) for info in metrics_info)
    ax.set_xlim(1, max_epoch)
    ax.margins(x=0)

    # Remove top/right spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Grid
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)

    # Legend beneath the plot
    n_metrics = len(metrics_info)
    ax.legend(
        title="",
        ncol=n_metrics,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        columnspacing=1.0,
    )

    # Adjust layout to make room for legend
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved multi-metric plot to {out_file}")
