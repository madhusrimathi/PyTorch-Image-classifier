import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.data import get_dataloaders
from src.model import SimpleCNN


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="models/cifar10_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_dataloaders(args.data_dir, args.batch_size)
    model = SimpleCNN(len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, loss_fn, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2%} | "
            f"test loss {test_loss:.4f} | test acc {test_acc:.2%}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": classes}, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
