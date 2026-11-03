"""Маленькая CNN для локальных синтетических кругов и квадратов."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1] / 'Данные/shapes'


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = correct = count = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    torch.manual_seed(42)
    tfm = transforms.ToTensor()
    loaders = {split: DataLoader(datasets.ImageFolder(ROOT / split, transform=tfm), batch_size=32, shuffle=(split == 'train')) for split in ('train', 'val', 'test')}
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 16 * 16, 2),
    )
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, best_epoch, best_weights = -1, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in loaders['train']:
            opt.zero_grad()
            loss_fn(model(images), labels).backward()
            opt.step()
        train_loss, train_acc = evaluate(model, loaders['train'], loss_fn)
        val_loss, val_acc = evaluate(model, loaders['val'], loss_fn)
        print(f'epoch={epoch:02d} train loss={train_loss:.3f} acc={train_acc:.3f}; validation loss={val_loss:.3f} acc={val_acc:.3f}')
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_val_loss):
            best_acc, best_val_loss, best_epoch, best_weights = val_acc, val_loss, epoch, deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    test_loss, test_acc = evaluate(model, loaders['test'], loss_fn)
    print(f'Best epoch={best_epoch}; final test loss={test_loss:.3f}, accuracy={test_acc:.3f}')
    print(f'Classes: {loaders["train"].dataset.classes}. Данные синтетические.')


if __name__ == '__main__':
    main()
