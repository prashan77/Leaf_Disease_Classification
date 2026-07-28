import time
import copy 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from data import get_dataloaders
from tqdm import tqdm

def build_model(pretrained: bool, num_classes=38) -> nn.Module:
    model = None 
    if pretrained:
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model 

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


def main(pretrained):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    variant, ratio, seed, batch_size, epochs = "color", 0.8, 42, 64, 5
    lr = 1e-4 if pretrained else 1e-3
    train_loader, val_loader, test_loader = get_dataloaders(variant, ratio, seed, batch_size, return_val=True, num_workers=4)
    model = build_model(pretrained, num_classes=38).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"train images: {len(train_loader.dataset):,}  val images: {len(val_loader.dataset):,}  test images: {len(test_loader.dataset):,}")

    best_state = None
    best_val_acc = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        if best_val_acc is None or best_val_acc < val_acc:
            best_state = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc
        print(f"epoch {epoch:>2}/{epochs}  train_loss={train_loss:.3f}   train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:.3f}  "
              f"({time.time() - t0:.1f}s)")
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"test_loss={test_loss:.3f}  test_acc={test_acc:.3f} ")

if __name__ == "__main__":
    main(True)
    main(False)