from project.training import train_utility_evaluator


# Train a position evaluation MLP on Tic-Tac-Toe
def eval_ttt():
    train_utility_evaluator("mnk_3_3_3")
