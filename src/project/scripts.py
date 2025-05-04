import torch
import torch.nn as nn
import torch.optim as optim

from project.data.utility import utility_dataframe
from project.models.mlp import MLPClassifier
from project.utils.datasets import tensor_dataset, data_loaders
from project.training import train_epoch, evaluate


def train_utility_evaluator():

    df = utility_dataframe(game="mnk_3_3_3")
    ds = tensor_dataset(df=df, label="utility")
    t_loader, v_loader = data_loaders(ds=ds, split=0.8, batch=64)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim=64, hidden_dims=[128, 64], num_classes=3)
    model.to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_epoch(model, t_loader, optimizer, criterion, dev)
        v_loss, v_acc = evaluate(model, v_loader, criterion, dev)

        print(
            f"Epoch {epoch:02d} "
            f"Train loss: {t_loss:.4f}, acc: {t_acc:.3f} | "
            f"Val loss: {v_loss:.4f}, acc: {v_acc:.3f}"
        )
