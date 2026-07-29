"""Edit these values to configure a training run."""

from pathlib import Path

# Experiment
ARCHITECTURE = "mobilenet_v2"
INITIALIZATION = "pretrained"
VARIANT = "color"
RATIO = 0.2
SEED = 42

# Training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2

# Hardware and output
NUM_WORKERS = 4
DEVICE = "auto"
AMP = True
OUTPUT_DIR: Path | None = None
OVERWRITE = False


def validate() -> None:
    """Check the values in this file before training starts."""

    if ARCHITECTURE not in {"mobilenet_v2", "vanilla_cnn"}:
        raise ValueError(f"unsupported architecture: {ARCHITECTURE}")
    if INITIALIZATION not in {"pretrained", "scratch"}:
        raise ValueError(f"unsupported initialization: {INITIALIZATION}")
    if ARCHITECTURE == "vanilla_cnn" and INITIALIZATION != "scratch":
        raise ValueError("vanilla_cnn only supports scratch initialization")
    if VARIANT not in {"color", "grayscale"}:
        raise ValueError(f"unsupported dataset variant: {VARIANT}")
    if RATIO not in {0.2, 0.5, 0.8}:
        raise ValueError("ratio must be 0.2, 0.5, or 0.8")
    if EPOCHS <= 0 or BATCH_SIZE <= 0 or LEARNING_RATE <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if not 0 <= WARMUP_EPOCHS < EPOCHS:
        raise ValueError("warmup epochs must be smaller than total epochs")
    if NUM_WORKERS < 0 or WEIGHT_DECAY < 0:
        raise ValueError("workers and weight decay cannot be negative")


def result_directory() -> Path:
    """Return the configured or automatically named results directory."""

    if OUTPUT_DIR is not None:
        return Path(OUTPUT_DIR)
    name = f"{ARCHITECTURE}_{VARIANT}_{INITIALIZATION}_{RATIO:g}_{SEED}"
    return Path("results") / name
