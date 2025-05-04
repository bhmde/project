import torch


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
