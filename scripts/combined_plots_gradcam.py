from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

figures = {
    "color": [
        (
            "Pretrained MobileNetV2",
            Path("results/mobilenet_v2_color_pretrained_0.2_42/gradcam"),
        ),
        (
            "Scratch MobileNetV2",
            Path("results/mobilenet_v2_color_scratch_0.2_42/gradcam"),
        ),
        (
            "Scratch Vanilla CNN",
            Path("results/vanilla_cnn_color_scratch_0.2_42/gradcam"),
        ),
    ],
    "grayscale": [
        (
            "Pretrained MobileNetV2",
            Path("results/mobilenet_v2_grayscale_pretrained_0.2_42/gradcam"),
        ),
        (
            "Scratch MobileNetV2",
            Path("results/mobilenet_v2_grayscale_scratch_0.2_42/gradcam"),
        ),
        (
            "Scratch Vanilla CNN",
            Path("results/vanilla_cnn_grayscale_scratch_0.2_42/gradcam"),
        ),
    ],
}

def find_pair(folder):
    originals = sorted(folder.glob("*_original.*"))

    for original in originals:
        stem = original.stem.removesuffix("_original")
        gradcam = folder / f"{stem}_gradcam.png"

        if gradcam.exists():
            return original, gradcam

    raise FileNotFoundError(f"No original/Grad-CAM pair found in {folder}")

column_titles = [
    "Correct: Original",
    "Correct: Grad-CAM",
    "Incorrect: Original",
    "Incorrect: Grad-CAM",
]

for variant, runs in figures.items():
    fig, axes = plt.subplots(
        len(runs),
        4,
        figsize=(20, 15),
        squeeze=False,
    )

    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=20)

    for row, (model_name, run_directory) in enumerate(runs):
        correct_original, correct_gradcam = find_pair(
            run_directory / "correct"
        )
        incorrect_original, incorrect_gradcam = find_pair(
            run_directory / "incorrect"
        )

        image_paths = [
            correct_original,
            correct_gradcam,
            incorrect_original,
            incorrect_gradcam,
        ]

        for column, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                axes[row, column].imshow(image.convert("RGB"))

            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

        axes[row, 0].set_ylabel(
            model_name,
            fontsize=19,
            fontweight="bold",
            labelpad=20,
        )

    fig.suptitle(
        f"Grad-CAM Comparison on {variant.title()} Images "
        "with 20% Training Data",
        fontsize=25,
    )

    fig.tight_layout(rect=(0.04, 0.02, 1, 0.95))

    fig.savefig(
        f"results/report_plots/{variant}_gradcam_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)