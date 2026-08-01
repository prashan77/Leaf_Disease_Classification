import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

runs = [
    (
        "MobileNetV2 Pretrained",
        Path("results/mobilenet_v2_color_pretrained_0.8_42/summary.json"),
    ),
    (
        "MobileNetV2 Scratch",
        Path("results/mobilenet_v2_color_scratch_0.8_42/summary.json"),
    ),
    (
        "Vanilla CNN Scratch",
        Path("results/vanilla_cnn_color_scratch_0.8_42/summary.json"),
    ),
]

class_names = json.loads(Path("data/classes.json").read_text())
display_ticks = np.arange(0, len(class_names), 2)

fig = plt.figure(figsize=(30, 10))

grid = fig.add_gridspec(
    1,
    4,
    width_ratios=[1, 1, 1, 0.045],
    left=0.05,
    right=0.94,
    bottom=0.12,
    top=0.84,
    wspace=0.04,
)

axes = [
    fig.add_subplot(grid[0, 0]),
    fig.add_subplot(grid[0, 1]),
    fig.add_subplot(grid[0, 2]),
]

colorbar_axis = fig.add_subplot(grid[0, 3])
image = None

for axis, (model_name, summary_path) in zip(axes, runs):
    summary = json.loads(summary_path.read_text())

    confusion = np.asarray(
        summary["test"]["confusion_matrix"],
        dtype=float,
    )

    row_totals = confusion.sum(axis=1, keepdims=True)

    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion),
        where=row_totals != 0,
    )

    image = axis.imshow(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )

    accuracy = summary["test"]["acc"]
    macro_f1 = summary["test"]["f1_macro"]

    axis.set_title(
        f"{model_name}\nAccuracy: {accuracy:.3f}, Macro-F1: {macro_f1:.3f}",
        fontsize=19,
        pad=10,
    )

    axis.set_xlabel(
        "Predicted Class Index",
        fontsize=16,
        labelpad=4,
    )

    axis.set_xticks(display_ticks)
    axis.set_yticks(display_ticks)
    axis.set_xticklabels(display_ticks, fontsize=11)
    axis.set_yticklabels(display_ticks, fontsize=11)
    axis.tick_params(axis="both", pad=2)

axes[0].set_ylabel(
    "True Class Index",
    fontsize=16,
    labelpad=4,
)

colorbar = fig.colorbar(
    image,
    cax=colorbar_axis,
)

colorbar.set_label(
    "Proportion of True-Class Images",
    fontsize=15,
    labelpad=4,
)

colorbar.ax.tick_params(
    labelsize=12,
    pad=3,
)

fig.suptitle(
    "Confusion Matrices for Models Trained on 80% of Color Images",
    fontsize=24,
    y=0.96,
)

fig.savefig(
    "color_80_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2,
)

plt.close(fig)