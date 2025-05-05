import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from project.data.utility import utility_dataframe
from project.utils.datasets import tensor_dataset
from project.activations import list_directory
from project.utils.checkpoints import models_directory, load_model_epoch
from project.models.mlp import MLPClassifier


def decode_board_states(state: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """
    Decode bit-encoded game states into board tensors.

    Args:
        state: Tensor of shape (B, 64) with values [-1, 1] representing bit values
        m: Number of rows in the game board
        n: Number of columns in the game board

    Returns:
        torch.Tensor: Tensor of shape (B, m, n) representing board states
    """
    # Convert from [-1, 1] to [0, 1]
    B = state.shape[0]
    bits = state.clone()
    bits[bits == -1] = 0
    turn = bits[:, -1]
    bits = bits[:, :-1]
    board_bits = bits[:, -m * n * 2 :].reshape(B, m, n, 2)
    board = torch.zeros(B, m, n)
    x_mask = board_bits[..., 0] == 1
    o_mask = board_bits[..., 1] == 1
    board[x_mask] = -1
    board[o_mask] = 1
    return board.to(int), turn


def draw_board(board, turn=None, outcome=None, ax=None):
    """
    Draw a game board visualization.

    Args:
        board: Tensor of shape (M, N) representing the board state
        turn: Optional scalar indicating whose turn it is (0 for X, 1 for O)
        outcome: Optional scalar for the game outcome (1=win, 0=tie, -1=loss)
        ax: Optional matplotlib axes object. If None, a new figure is created.

    Returns:
        matplotlib.axes.Axes: The axes object containing the visualization
    """
    m, n = board.shape

    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Create a custom colormap for the board values
    cmap = plt.cm.colors.ListedColormap(["lightgray", "lightblue", "salmon"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Use imshow to display the board with proper extent to align grid with cell centers
    ax.imshow(
        board, cmap=cmap, norm=norm, extent=[-0.5, n - 0.5, m - 0.5, -0.5]
    )

    # Add cell values as text
    for i in range(m):
        for j in range(n):
            cell_value = board[i, j].item()
            cell_text = (
                "O" if cell_value == 1 else "X" if cell_value == -1 else "·"
            )
            ax.text(
                j,
                i,
                cell_text,
                ha="center",
                va="center",
                color="black",
                fontsize=20,
                fontweight="bold",
            )

    # Add properly centered grid lines
    # Vertical lines
    for i in range(n - 1):
        ax.axvline(i + 0.5, color="black", linestyle="-", linewidth=1.5)

    # Horizontal lines
    for i in range(m - 1):
        ax.axhline(i + 0.5, color="black", linestyle="-", linewidth=1.5)

    # Add border around the entire grid
    ax.plot(
        [-0.5, n - 0.5, n - 0.5, -0.5, -0.5],
        [-0.5, -0.5, m - 0.5, m - 0.5, -0.5],
        color="black",
        linewidth=2,
    )

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title indicating current player's turn and outcome
    title = []

    if turn is not None:
        title.append(f"Player: {'X' if turn == 0 else 'O'}")

    if outcome is not None:
        outcome_text = {1: "Win", 0: "Tie", -1: "Loss"}.get(
            int(outcome), "Unknown"
        )
        title.append(f"Outcome: {outcome_text}")

    if title:
        ax.set_title(" | ".join(title), fontsize=16)

    # Ensure square cells
    ax.set_aspect("equal")

    return ax


def visualize_grid_of_boards(boards, turns=None, outcomes=None):
    """
    Visualize a sequence of game boards in a grid layout.

    Args:
        boards: List of board tensors or single tensor with first dimension as sequence
        turns: Optional list of turn indicators
        outcomes: Optional list of game outcomes (1=win, 0=tie, -1=loss)
    """
    if isinstance(boards, torch.Tensor) and boards.dim() == 3:
        # Convert to list of boards
        boards = [boards[i] for i in range(boards.shape[0])]

    n_boards = len(boards)
    n_cols = min(5, n_boards)  # Maximum 5 boards per row
    n_rows = (n_boards + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Convert to numpy array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, board in enumerate(boards):
        row = i // n_cols
        col = i % n_cols

        turn = turns[i] if turns is not None and i < len(turns) else None
        outcome = (
            outcomes[i] if outcomes is not None and i < len(outcomes) else None
        )

        draw_board(board, turn, outcome, axes[row, col])

    # Hide unused subplots
    for i in range(n_boards, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig, axes


def run_pca_on_activations(activation_df, n_components=2, act_prefix="act_"):
    """
    Run PCA on activation columns of a DataFrame.

    Parameters:
    -----------
    activation_df : pandas.DataFrame
        DataFrame containing activation columns
    n_components : int, default=2
        Number of principal components to compute
    act_prefix : str, default='act_'
        Prefix of column names to select for PCA

    Returns:
    --------
    tuple: (pca_result, pca_model, activation_columns)
        - pca_result: DataFrame with principal components
        - pca_model: fitted PCA model
        - activation_columns: list of column names used for PCA
    """
    # Get the activation columns (those starting with act_prefix)
    activation_columns = [
        col for col in activation_df.columns if col.startswith(act_prefix)
    ]

    # Extract activation data
    X = activation_df[activation_columns].values

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(
        data=pca_result, columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return pca_df, pca, activation_columns


def plot_pca(
    pca_df,
    activation_df,
    pca_model,
    color_by="utility",
    title="PCA of Neural Network Activations",
    ax=None,
):
    """
    Plot PCA results with color coding based on a parameter.

    Parameters:
    -----------
    pca_df : pandas.DataFrame
        DataFrame containing principal components
    activation_df : pandas.DataFrame
        Original DataFrame containing activation and other columns
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model
    color_by : str, default='utility'
        Column name to use for color coding the points
    title : str, default="PCA of Neural Network Activations"
        Title for the plot
    ax : matplotlib.axes.Axes, default=None
        Optional matplotlib axes object. If None, a new figure is created.
    """
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
    else:
        fig = ax.figure

    # Check if color_by column exists in the DataFrame
    if color_by in activation_df.columns:
        # Get color values
        colors = activation_df[color_by].values

        # Scatter plot of first two principal components with color coding
        scatter = ax.scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            c=colors,
            cmap="viridis",
            alpha=0.7,
            edgecolors="w",
            linewidth=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_by)
    else:
        # Fallback if column doesn't exist
        scatter = ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)
        print(
            f"Warning: Column '{color_by}' not found. Plotting without color coding."
        )

    # Add labels and title
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%})")
    ax.set_title(title)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig, ax


def run_visualizations(args):
    ### mwe of board visualization. proper feature vis next

    df = utility_dataframe(game=args.game)

    # Correctly extract m, n, k from args.game (format: "mnk_3_3_3")
    parts = args.game.split("_")
    m = int(parts[1])  # First number after "mnk_"
    n = int(parts[2])  # Second number

    dataset = tensor_dataset(df=df, label="utility")

    # # Example of a single board visualization
    # x, y = dataset[20]
    # print(x)
    # print(x.shape)
    # print(f"Outcome: {y.item()}")

    # # Decode the game state
    # board, turn = decode_board_states(x.unsqueeze(0), m, n)
    # board = board.squeeze(0)
    # turn = turn.squeeze(0)

    # # Squeeze to remove batch dimension for visualization
    # print("board.shape", board.shape)
    # print("turn.shape", turn.shape)

    # # Use the modular draw_board function with outcome
    # fig, ax = plt.subplots(figsize=(8, 8))
    # draw_board(board, turn, y.item(), ax)
    # plt.tight_layout()

    # # Visualize a grid of random boards with their outcomes
    # n_examples = 20
    # indices = torch.randint(0, len(dataset), (n_examples,))
    # x, y = dataset[indices]
    # boards, turns = decode_board_states(x, m, n)

    # # Pass outcomes (y values) to the visualization function
    # visualize_grid_of_boards(boards, turns, y)

    activation_dfs = {}
    path = f"{models_directory}/{args.model}/{args.game}"
    epochs = list_directory(path)
    model = MLPClassifier(input_dim=64, num_classes=3)
    for epoch in epochs:
        activation_dfs[int(epoch)] = pd.read_pickle(
            f"models/{model.name()}/{args.game}/{epoch}/activations.pkl"
        )

        activation_cols = [c for c in df.columns if c.startswith("act")]
        X = df[activation_cols].values  # shape (N, D)

    for interpretable_feature in [
        # 'remoteness',
        # 'orbit_rep',
        # 'fork_exists',
        # 'ply',
        # 'center_control',
        # 'corner_count',
        # 'edge_count',
        "utility"
    ]:
        # Get sorted epochs (newest first)
        sorted_epochs = reversed(sorted(activation_dfs.keys()))
        epochs_list = list(sorted_epochs)
        n_epochs = len(epochs_list)

        # Create a grid of subplots for each epoch
        n_cols = min(3, n_epochs)  # Max 3 plots per row
        n_rows = (n_epochs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows)
        )
        fig.suptitle(f"{interpretable_feature}", fontsize=16)

        # Ensure axes is a 2D array for consistent indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Create a PCA plot for each epoch
        for i, epoch in enumerate(epochs_list):
            row = i // n_cols
            col = i % n_cols

            activation_df = activation_dfs[epoch]
            pca_df, pca, activation_columns = run_pca_on_activations(
                activation_df
            )

            plot_pca(
                pca_df,
                activation_df,
                pca,
                color_by=interpretable_feature,
                title=f"Epoch {epoch}",
                ax=axes[row, col],
            )

        # Hide unused subplots if any
        for i in range(n_epochs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        # Save figure to file
        plt.savefig(
            f"reports/visualization/pca_{interpretable_feature}_epochs.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
