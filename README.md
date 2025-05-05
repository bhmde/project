# Mechanistic Interpretability of Learning Dynamics in Game-Playing Neural Networks

Bradley Arias, Max Fierro, Humberto Gutierrez, Dmytro Krukovskyi, Elliot Meldrum

## Abstract

We investigate the relationship between interpretability and learning dynamics within neural networks that optimize for two computationally equivalent game-theoretic tasks; decision-making and utility evaluation. In the decision-making network, we use a selection of probes to determine the recoverability of interpretable features over the course of training. In the evaluation network, we analyze the preservation of algebraic properties of the native state representations across training epochs through dimensionality reduction and mechanistic analysis of intermediate activations. We leverage our access to a strong solution of the game underlying these tasks to eschew the need for human selection of interpretable features and to provide ground-truth statistics which help with the interpretation of our measurements.

## Development 

This project uses [`uv`](https://github.com/astral-sh/uv). To install:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The tool [`black`](https://github.com/psf/black) is used for formatting. To install with `uv`:

```
uv tool install black
```

Then, to format your changes with `black`:

```
uv run black .
```

To run with `uv`, first ensure it has installed all the packages you will need:

```
uv sync
```

Then execute a script (see below for options):

```
uv run <script>
```

## Scripts

1. `train-evaluator`: Train a position evaluation MLP on a game, saving a progression of model checkpoints throughout training.

2. `gen-activations`: For each model checkpoint created during `train-evaluator`, generate a dataframe of its activations at a select layer on different datapoints together with the datapoint (game state) which generated those activations (and the interpretable features of that state).

3. [TODO] `train-probes`: For each dataframe generated during `gen-activations`, train a linear probe to predict select features associated with the input state that generated those activations. 

4. [TODO] `feature-visualization`: Visualize the learned features...
