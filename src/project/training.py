import torch
import torch.nn as nn
import torch.optim as optim

from project.data.utility import utility_dataframe
from project.models.mlp import MLPClassifier
from project.utils.datasets import tensor_dataset, data_loaders
from project.utils.checkpoints import save_model_epoch


def train_utility_evaluator(game: str):

    df = utility_dataframe(game=game)
    ds = tensor_dataset(df=df, label="utility")
    t_loader, v_loader = data_loaders(ds=ds, split=0.8, batch=64)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim=64, num_classes=3)
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

        if epoch % 2 == 0:
            save_model_epoch(
                epoch=epoch, game=game, name=MLPClassifier.name(), model=model
            )


# ----------------
# HELPER FUNCTIONS
# ----------------


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        if logits.ndim == 1 or logits.size(1) == 1:
            preds = (logits > 0).long()
        else:
            preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item() * X.size(0)
            if logits.ndim == 1 or logits.size(1) == 1:
                preds = (logits > 0).long()
            else:
                preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return running_loss / total, correct / total
