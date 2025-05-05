# Script for debugging the gen-activations

from activations import generate_model_activations
from models.mlp import MLPClassifier

if __name__ == "__main__":
    game = "mnk_3_3_3"
    model = MLPClassifier.name()
    generate_model_activations(game, model)