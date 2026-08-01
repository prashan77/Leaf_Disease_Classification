import csv
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
    wspace=0.08,
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
    "results/report_plots/color_80_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2,
)

plt.close(fig)

def clean_name(name):
    return name.replace("___", " - ").replace("_", " ")

table_rows = []

for model_name, summary_path in runs:
    summary = json.loads(summary_path.read_text())

    confusion = np.asarray(
        summary["test"]["confusion_matrix"],
        dtype=int,
    )

    model_errors = []

    for true_index in range(len(class_names)):
        true_total = int(confusion[true_index].sum())

        for predicted_index in range(len(class_names)):
            if true_index == predicted_index:
                continue

            error_count = int(
                confusion[true_index, predicted_index]
            )

            if error_count == 0:
                continue

            error_rate = (
                error_count / true_total
                if true_total > 0
                else 0
            )

            model_errors.append(
                {
                    "model": model_name,
                    "true_class": clean_name(
                        class_names[true_index]
                    ),
                    "predicted_class": clean_name(
                        class_names[predicted_index]
                    ),
                    "error_count": error_count,
                    "true_class_total": true_total,
                    "error_rate_percent": error_rate * 100,
                }
            )

    model_errors.sort(
        key=lambda row: (
            row["error_rate_percent"],
            row["error_count"],
        ),
        reverse=True,
    )

    table_rows.extend(model_errors[:5])

output_csv = Path("results/report_plots/top_color_confusions.csv")

with output_csv.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "model",
            "true_class",
            "predicted_class",
            "error_count",
            "true_class_total",
            "error_rate_percent",
        ],
    )

    writer.writeheader()

    for row in table_rows:
        csv_row = row.copy()
        csv_row["error_rate_percent"] = (
            f"{row['error_rate_percent']:.2f}"
        )
        writer.writerow(csv_row)

print(
    f"{'Model':<27} "
    f"{'True class':<40} "
    f"{'Predicted class':<40} "
    f"{'Count':>7} "
    f"{'Rate':>9}"
)

print("-" * 130)

for row in table_rows:
    print(
        f"{row['model']:<27} "
        f"{row['true_class']:<40} "
        f"{row['predicted_class']:<40} "
        f"{row['error_count']:>7} "
        f"{row['error_rate_percent']:>8.2f}%"
    )

print()
print("Created color_80_confusion_matrices.png")
print(f"Created {output_csv}")