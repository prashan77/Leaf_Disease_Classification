"""High-level orchestration for one controlled training run."""

from __future__ import annotations

import json
import time

import torch
from torch.optim import AdamW

from src import config as settings
from src.data import class_names, get_dataloaders
from src.metrics import plot_metrics
from src.models import build_model
from src.training.engine import (
    build_scheduler,
    evaluate,
    seed,
    select_device,
    synchronize,
    train_one_epoch,
)
from src.training.results import RunArtifacts


def run_experiment() -> dict[str, object]:
    """Train, select the best checkpoint, and evaluate it once on test."""

    settings.validate()
    seed(settings.SEED)
    device = select_device(settings.DEVICE)
    use_amp = settings.AMP and device.type in {"cuda", "mps"}
    names = class_names()
    num_classes = len(names)
    artifacts = RunArtifacts(settings.result_directory(), settings.OVERWRITE)

    train_loader, val_loader, test_loader = get_dataloaders(
        settings.VARIANT,
        settings.RATIO,
        settings.SEED,
        settings.BATCH_SIZE,
        return_val=True,
        num_workers=settings.NUM_WORKERS,
    )
    model = build_model(
        settings.ARCHITECTURE,
        settings.INITIALIZATION,
        num_classes,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=settings.LEARNING_RATE,
        weight_decay=settings.WEIGHT_DECAY,
    )
    scheduler = build_scheduler(
        optimizer,
        settings.EPOCHS,
        settings.WARMUP_EPOCHS,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = {
        "architecture": settings.ARCHITECTURE,
        "variant": settings.VARIANT,
        "initialization": settings.INITIALIZATION,
        "ratio": settings.RATIO,
        "seed": settings.SEED,
        "batch_size": settings.BATCH_SIZE,
        "epochs": settings.EPOCHS,
        "learning_rate": settings.LEARNING_RATE,
        "weight_decay": settings.WEIGHT_DECAY,
        "warmup_epochs": settings.WARMUP_EPOCHS,
        "num_workers": settings.NUM_WORKERS,
        "requested_device": settings.DEVICE,
        "requested_amp": settings.AMP,
        "device": str(device),
        "amp": use_amp,
        "num_classes": num_classes,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
    }
    artifacts.write_config(config)
    print(json.dumps(config, indent=2))

    best_f1 = -1.0
    best_epoch = 0
    train_accuracy = []
    val_accuracy = []
    val_f1 = []
    training_times = []
    for epoch in range(1, settings.EPOCHS + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        synchronize(device)
        epoch_started = time.perf_counter()
        training_started = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
        )
        synchronize(device)
        training_seconds = time.perf_counter() - training_started
        validation = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            use_amp,
            description=f"epoch {epoch} val",
        )
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        train_accuracy.append(train_acc)
        val_accuracy.append(validation["acc"])
        val_f1.append(validation["f1_macro"])
        training_times.append(training_seconds)

        result = {
            "architecture": settings.ARCHITECTURE,
            "variant": settings.VARIANT,
            "initialization": settings.INITIALIZATION,
            "ratio": settings.RATIO,
            "seed": settings.SEED,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": validation["loss"],
            "val_acc": validation["acc"],
            "val_f1_macro": validation["f1_macro"],
            "val_f1_weighted": validation["f1_weighted"],
            "training_seconds": training_seconds,
            "val_inference_seconds": validation["inference_seconds"],
            "epoch_seconds": epoch_seconds,
        }
        artifacts.append_epoch(result)

        if validation["f1_macro"] > best_f1:
            best_f1 = validation["f1_macro"]
            best_epoch = epoch
            artifacts.save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_f1,
                config,
                names,
            )

        print(
            f"epoch={epoch:02d} lr={learning_rate:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={validation['loss']:.4f} val_acc={validation['acc']:.4f} "
            f"val_f1={validation['f1_macro']:.4f} seconds={epoch_seconds:.1f}"
        )
        scheduler.step()

    artifacts.load_model(model, device)
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        num_classes,
        use_amp,
        description="final test",
    )
    plot_name = " ".join(
        (
            settings.ARCHITECTURE.replace("_", " ").title(),
            settings.INITIALIZATION.title(),
            settings.VARIANT.title(),
            f"{int(settings.RATIO * 100)} Percent",
        )
    )
    plot_paths = plot_metrics(
        train_accuracy,
        val_accuracy,
        val_f1,
        test_metrics["confusion_matrix"],
        plot_name,
        output_dir=artifacts.root / "plots",
    )
    test_results = {
        **test_metrics,
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    }
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1,
        "training_seconds": sum(training_times),
        "test": test_results,
        "checkpoint": str(artifacts.checkpoint_path),
        "plots": [str(path) for path in plot_paths],
    }
    artifacts.write_summary(summary)
    print(f"\nbest epoch: {best_epoch}")
    print(
        json.dumps(
            {
                name: value
                for name, value in test_metrics.items()
                if name != "confusion_matrix"
            },
            indent=2,
        )
    )
    print(f"results written to {artifacts.root}")
    return summary
