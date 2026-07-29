"""
A plain CNN, no pretrained weights, no inverted residuals -- the kind of
architecture MobileNetV2 is implicitly being compared against in spirit

Uses get_dataloaders() from src/data.py, so it inherits the same leaf-grouped,
class-stratified, nested splits as every other run -- no separate data path.
"""

from __future__ import annotations

import time
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders
from tqdm import tqdm

class VanillaCNN(nn.Module):
    """
    Five conv blocks (BN + ReLU + MaxPool), global average pool, one FC layer.
    No residual connections, no depthwise separable convs -- a baseline architecture, 
    as opposed to MobileNetV2 trained from scratch.

    Input: (B, 3, 224, 224) -- matches the frozen get_dataloaders() contract.
    Output: (B, num_classes) logits.
    """

    def __init__(self, num_classes: int = 30, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),     
            block(16, 32),    
            block(32, 64),   
            block(64, 128),  
            block(128, 128),  
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_one_epoch(model, loader, opt, device) -> float:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct/ n


def compute_f1(confusion_matrix: torch.Tensor) -> float:
    true_positive = confusion_matrix.diag()
    precision = true_positive / confusion_matrix.sum(0).clamp(min=1)
    recall = true_positive / confusion_matrix.sum(1).clamp(min=1)
    return (2 * precision * recall / (precision + recall).clamp(min=1e-12)).mean().item()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model, loader, device, num_classes=30):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    synchronize(device)
    start = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        predictions = logits.argmax(1)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (predictions == y).sum().item()
        indices = (y * num_classes + predictions).detach().cpu()
        confusion_matrix += torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        n += x.size(0)
    synchronize(device)
    inference_time = time.perf_counter() - start
    return total_loss / n, correct / n, compute_f1(confusion_matrix), confusion_matrix, inference_time


def plot_metrics(train_accuracy, val_accuracy, val_f1, confusion_matrix, name):
    output_dir = Path("images")
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_accuracy) + 1)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(epochs, train_accuracy, marker="o", label="Train")
    axis.plot(epochs, val_accuracy, marker="o", label="Validation")
    axis.set_title(f"{name} Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.legend()
    figure.tight_layout()
    file_name = name.lower().replace(" ", "_")
    figure.savefig(output_dir / f"{file_name}_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(epochs, val_f1, marker="o")
    axis.set_title(f"{name} Validation F1")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("F1")
    figure.tight_layout()
    figure.savefig(output_dir / f"{file_name}_validation_f1.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(confusion_matrix.numpy(), cmap="Blues")
    axis.set_title(f"{name} Confusion Matrix")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / f"{file_name}_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    variant, ratio, seed, batch_size, epochs, lr = "color", 0.8, 42, 64, 5, 1e-3

    train_loader, val_loader, test_loader = get_dataloaders(variant, ratio, seed, batch_size, return_val=True, num_workers=2)
    model = VanillaCNN(num_classes=30).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"params: {model.num_params():,}")
    print(f"train images: {len(train_loader.dataset):,}  val images: {len(val_loader.dataset):,}   test images: {len(test_loader.dataset):,}")

    best_state = None
    best_val_acc = None
    train_accuracy, val_accuracy, val_f1, training_time = [], [], [], []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device)
        epoch_time = time.time() - t0
        val_loss, val_acc, epoch_f1, _, _ = evaluate(model, val_loader, device)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        val_f1.append(epoch_f1)
        training_time.append(epoch_time)
        if best_val_acc is None or best_val_acc < val_acc:
            best_state = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc
        print(f"epoch {epoch:>2}/{epochs}  train_loss={train_loss:.3f}  train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:.3f}  val_f1={epoch_f1:.3f}  "
              f"({epoch_time:.1f}s)")
    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, confusion_matrix, inference_time = evaluate(model, test_loader, device)
    plot_metrics(train_accuracy, val_accuracy, val_f1, confusion_matrix, "Vanilla CNN")
    print(f"test_loss={test_loss:.3f}  test_acc={test_acc:.3f}  test_f1={test_f1:.3f}  "
          f"training_time={sum(training_time):.3f}s  inference_time={inference_time:.3f}s")

if __name__ == "__main__":
    main()
