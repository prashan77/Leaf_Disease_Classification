import csv
from pathlib import Path

import matplotlib.pyplot as plt

results_root = Path("results")
ratios = [0.2, 0.5, 0.8]

styles = {
    ("mobilenet_v2", "pretrained"): {
        "label": "MobileNetV2 Pretrained",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
    },
    ("mobilenet_v2", "scratch"): {
        "label": "MobileNetV2 Scratch",
        "color": "#d62728",
        "linestyle": "--",
        "marker": "s",
    },
    ("vanilla_cnn", "scratch"): {
        "label": "Vanilla CNN Scratch",
        "color": "#2ca02c",
        "linestyle": "-.",
        "marker": "^",
    },
}

runs = []

for history_path in sorted(results_root.glob("*/history.csv")):
    with history_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        continue

    first = rows[0]

    runs.append(
        {
            "architecture": first["architecture"],
            "initialization": first["initialization"],
            "variant": first["variant"],
            "ratio": float(first["ratio"]),
            "epochs": [int(row["epoch"]) for row in rows],
            "f1": [float(row["val_f1_macro"]) for row in rows],
        }
    )

for variant in ["color", "grayscale"]:
    fig = plt.figure(figsize=(18, 13))

    grid = fig.add_gridspec(
        2,
        4,
        left=0.08,
        right=0.98,
        top=0.91,
        bottom=0.14,
        wspace=0.55,
        hspace=0.42,
    )

    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]

    for axis, ratio in zip(axes, ratios):
        matching_runs = [
            run
            for run in runs
            if run["variant"] == variant and run["ratio"] == ratio
        ]

        for run in matching_runs:
            key = (run["architecture"], run["initialization"])
            style = styles.get(key)

            if style is None:
                continue

            axis.plot(
                run["epochs"],
                run["f1"],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=3,
                markersize=7,
                markeredgewidth=1,
                markeredgecolor="white",
            )

        axis.set_title(
            f"{int(ratio * 100)}% Training Data",
            fontsize=20,
        )
        axis.set_xlabel("Epoch", fontsize=18)
        axis.set_ylabel("Validation Macro-F1", fontsize=18)
        axis.set_xlim(1, 20)
        axis.set_ylim(0, 1.02)
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(True, alpha=0.3)

    handles = []
    labels = []

    for axis in axes:
        current_handles, current_labels = axis.get_legend_handles_labels()

        for handle, label in zip(current_handles, current_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=17,
        bbox_to_anchor=(0.5, 0.035),
    )

    fig.suptitle(
        f"Validation Macro-F1 on {variant.title()} Images",
        fontsize=25,
    )

    fig.savefig(
        f"results/report_plots/{variant}_validation_f1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)