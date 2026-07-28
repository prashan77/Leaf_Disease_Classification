"""

A plain CNN, no pretrained weights, no inverted residuals -- the kind of
architecture MobileNetV2 is implicitly being compared against in spirit

Uses get_dataloaders() from src/data.py, so it inherits the same leaf-grouped,
class-stratified, nested splits as every other run -- no separate data path.
"""

from __future__ import annotations

import time
import copy

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

    def __init__(self, num_classes: int = 38, dropout: float = 0.3):
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
    total_loss, n = 0.0, 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    variant, ratio, seed, batch_size, epochs, lr = "color", 0.8, 42, 64, 5, 1e-3

    train_loader, val_loader, test_loader = get_dataloaders(variant, ratio, seed, batch_size, return_val=True, num_workers=2)
    model = VanillaCNN(num_classes=38).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"params: {model.num_params():,}")
    print(f"train images: {len(train_loader.dataset):,}  val images: {len(val_loader.dataset):,}   test images: {len(test_loader.dataset):,}")

    best_state = None
    best_val_acc = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, opt, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        if best_val_acc is None or best_val_acc < val_acc:
            best_state = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc
        print(f"epoch {epoch:>2}/{epochs}  train_loss={train_loss:.3f}  "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:.3f}  "
              f"({time.time() - t0:.1f}s)")
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"test_loss={test_loss:.3f}  test_acc={test_acc:.3f} ")

if __name__ == "__main__":
    main()
