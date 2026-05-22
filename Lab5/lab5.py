import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data"
NUM_CLASSES = 100
BATCH_SIZE = 128

TRAIN_PART = 1.0

EXPERIMENTS = [
    {"name": "Exp 1", "channels": [64, 128, 256], "dropout": 0.30,
     "optimizer": "adam", "lr": 0.001, "weight_decay": 1e-4, "pool": "max", "epochs": 35},

    {"name": "Exp 2", "channels": [64, 128, 256, 512], "dropout": 0.35,
     "optimizer": "adam", "lr": 0.001, "weight_decay": 1e-4, "pool": "max", "epochs": 40},

    {"name": "Exp 3", "channels": [64, 128, 256, 512], "dropout": 0.40,
     "optimizer": "adamw", "lr": 0.001, "weight_decay": 1e-4, "pool": "max", "epochs": 45},

    {"name": "Exp 4", "channels": [96, 192, 384, 512], "dropout": 0.40,
     "optimizer": "adamw", "lr": 0.0008, "weight_decay": 2e-4, "pool": "avg", "epochs": 45},

    {"name": "Exp 5", "channels": [96, 192, 384, 512], "dropout": 0.45,
     "optimizer": "sgd", "lr": 0.05, "weight_decay": 5e-4, "pool": "max", "epochs": 50}
]


def get_loaders():
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform_train
    )

    test_set = torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform_test
    )

    if TRAIN_PART < 1.0:
        size = int(len(train_set) * TRAIN_PART)
        train_set, _ = torch.utils.data.random_split(train_set, [size, len(train_set) - size])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, pool_type):
        super().__init__()

        pool = nn.MaxPool2d(2) if pool_type == "max" else nn.AvgPool2d(2)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            pool,
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.block(x)

class CNNClassifier(nn.Module):
    def __init__(self, channels, dropout, pool_type):
        super().__init__()

        blocks = []
        in_channels = 3

        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels, dropout, pool_type))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def create_model(cfg):
    return CNNClassifier(
        channels=cfg["channels"],
        dropout=cfg["dropout"],
        pool_type=cfg["pool"]
    ).to(DEVICE)

def create_optimizer(model, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"],
            nesterov=True
        )

    raise ValueError("Unknown optimizer")

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count

def train_experiment(cfg, train_loader, test_loader):
    model = create_model(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, cfg)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0
    early_stop_patience = 8

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": []
    }

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion)

        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)

        print(
            f'{cfg["name"]} | epoch {epoch:02d}/{cfg["epochs"]} | '
            f'train loss={train_loss:.4f}, acc={train_acc:.4f} | '
            f'test loss={test_loss:.4f}, acc={test_acc:.4f}'
        )

        if test_loss < best_loss:
            best_loss = test_loss
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    final_loss, final_acc = run_epoch(model, test_loader, criterion)

    return model, history, final_loss, final_acc

def architecture_description(cfg):
    parts = []

    in_ch = 3
    for i, out_ch in enumerate(cfg["channels"], start=1):
        parts.append(
            f"ConvBlock {i}: Conv2D({in_ch}->{out_ch}, 3x3, padding=1), "
            f"BatchNorm, ReLU, Conv2D({out_ch}->{out_ch}), BatchNorm, ReLU, "
            f"{cfg['pool'].capitalize()}Pool2D, Dropout2D({cfg['dropout']})"
        )
        in_ch = out_ch

    parts.append(f"Dense(512, ReLU), Dropout({cfg['dropout']})")
    return "; ".join(parts)

def plot_history(history, name):
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["test_loss"], label="test loss")
    plt.title(f"Loss — {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.grid()
    plt.legend()
    plt.savefig(f"{name}_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["test_accuracy"], label="test accuracy")
    plt.title(f"Accuracy — {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.savefig(f"{name}_accuracy.png", dpi=150)
    plt.close()

def main():
    print("Device:", DEVICE)

    train_loader, test_loader = get_loaders()

    results = []
    best_model = None
    best_history = None
    best_name = None
    best_accuracy = 0.0

    for index, cfg in enumerate(EXPERIMENTS, start=1):
        print("\n" + "=" * 70)
        print(cfg["name"])
        print("=" * 70)

        model, history, test_loss, test_acc = train_experiment(cfg, train_loader, test_loader)

        results.append({
            "index": index,
            "config": cfg,
            "epochs": len(history["train_loss"]),
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_history = history
            best_name = cfg["name"]

    print("\nRESULTS TABLE")
    print("№ | epochs | hidden layers | optimizer | test loss | test accuracy")

    for row in results:
        cfg = row["config"]
        print(
            f'{row["index"]} | {row["epochs"]} | '
            f'{architecture_description(cfg)} | '
            f'{cfg["optimizer"]} | '
            f'{row["test_loss"]:.4f} | '
            f'{row["test_accuracy"]:.4f}'
        )

    print("\nBest experiment:", best_name)

    plot_history(best_history, best_name)
    torch.save(best_model.state_dict(), "best_cifar100_cnn_pytorch.pt")
    print("Best model saved as best_cifar100_cnn_pytorch.pt")

if __name__ == "__main__":
    main()
