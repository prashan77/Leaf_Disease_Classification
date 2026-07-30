# Leaf_Disease_Classification

CS7643 final project: does ImageNet pretraining help MobileNetV2 classify
plant leaf disease, versus training the same architecture from scratch?
Compared against a `VanillaCNN` control and, historically, Mohanty et al.'s
AlexNet/GoogLeNet baselines on the same [PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage)
dataset. Uses a leaf-grouped, class-stratified split protocol so augmented
copies of the same physical leaf never straddle train/test.

## Repo layout

```
scripts/
  download_data.py   fetches data.zip + leaf-map.json from HF (network, one time)
  prepare_data.py     data.zip + leaf-map -> data/raw/, manifest.csv, classes.json
  make_splits.py       manifest.csv -> splits/*.json (train/val/test index files)
  grad_cam.py           Grad-CAM overlay for specific images, given a checkpoint
  grad_cam_sample.py    Grad-CAM over a random sample of correct/incorrect test predictions
src/
  config.py           edit these values to configure the next training run
  main.py             entry point: `python -m src.main` trains with config.py's settings
  interfaces.py        FROZEN -- get_dataloaders/build_model/compute_metrics signatures
  data.py               PlantVillageDataset + get_dataloaders(), classes.json-driven labels
  models/               MobileNetV2 and VanillaCNN, both built via build_model()
  training/             engine.py (train/eval loops), experiment.py (orchestration),
                         results.py (checkpoint + history + config persistence)
  metrics/               f1_from_confusion(), plot_metrics() (accuracy/F1/confusion-matrix plots)
data/
  manifest.csv          committed -- one row per kept image per variant, with leaf_id
  classes.json           committed -- frozen label ordering -> int index (30 classes, see below)
  class_counts.csv       committed -- per-class image/leaf counts, for macro-F1 weighting
  data.zip, raw/, leaf_grouping/   gitignored -- regenerate locally, see below
splits/
  {variant}_{ratio}_{seed}.json   committed -- e.g. color_0.8_42.json
results/
  {architecture}_{variant}_{initialization}_{ratio}_{seed}/
    config.json, history.csv, best_model.pt, summary.json, plots/, gradcam/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda, using the pinned `environment.yml`:
```bash
conda env create -f environment.yml
conda activate leaf-disease-classification
```

Both pin identical versions: `torch` 2.13.0, `torchvision` 0.28.0, `numpy`
2.5.1, `Pillow` 12.3.0, `matplotlib` 3.11.1, `tqdm` 4.70.0, `huggingface_hub`
1.24.0.

> Venvs embed absolute paths (`.venv/bin/activate`, etc.) and are **not**
> relocatable -- if you move the project folder, delete and recreate `.venv`
> rather than trying to keep the old one.

## Rebuilding the data pipeline

The raw images and leaf map are gitignored (too large / not ours to
redistribute), so anyone starting fresh needs to regenerate `data/raw/`, then
the splits:

```bash
# 1. fetch the zip + leaf map (network, one time)
python -m scripts.download_data --root data

# 2. extract + build manifest.csv / classes.json / class_counts.csv (offline)
python -m scripts.prepare_data --root data

# 3. build the leaf-grouped, class-stratified, nested splits (offline)
python -m scripts.make_splits --root data --ratios 0.8 0.5 0.2 --seed 42
```

Steps 2 and 3 each print a sha256 digest of what they produce (`manifest.csv`,
each `splits/*.json`). Everyone on the team should get identical digests -- if
yours differs, something upstream (zip version, leaf map, seed) doesn't match.

`prepare_data.py --variants` defaults to `color grayscale`; pass
`--variants color grayscale segmented` if you also need the segmented variant
(present in the zip, not extracted by default).

## Why classes.json has 30 classes, not 38

PlantVillage ships 38 classes, but `leaf-map.json` (the leaf-identity data
that makes the leak-safe split possible) has **zero** verified coverage for 8
of them: all 4 `Corn_(maize)___*` classes, `Grape___healthy`,
`Squash___Powdery_mildew`, `Tomato___Target_Spot`, and
`Tomato___Tomato_mosaic_virus`. A class with no verified leaf identity at all
can never produce a leak-safe split no matter what `make_splits.py` does
downstream, so `prepare_data.py` drops those 8 classes entirely -- not just
from the default split, but from `classes.json`, `class_counts.csv`, and
`manifest.csv` -- rather than silently keeping a 38-class label space where 8
classes have no usable data. `build_model`'s `num_classes` default (in both
`src/interfaces.py` and the model constructors) is `30` to match. See "Known
limitation: leaf-map coverage" below for the partial-coverage classes that
did survive with fewer images.

## Split design (`scripts/make_splits.py`)

Three properties, chosen deliberately:

- **Leaf-grouped** -- by default, only images with a verified `leaf_id` are
  included, so every known photo of one physical leaf lands on the same side
  of the split. PlantVillage shoots each leaf multiple times; a random
  image-level split would leak near-duplicates into test and inflate every
  reported number.
- **Class-stratified** -- the shuffle-and-cut happens within each class, not
  globally, since class sizes range ~150-5500 images (36x imbalance). A global
  cut can starve a rare class's train or test set.
- **Nested** -- one shuffle per class, reused across ratios, so
  `train(0.2) ⊂ train(0.5) ⊂ train(0.8)`. The accuracy-vs-training-size curve
  then varies exactly one thing (how much data), not which leaves got drawn.

Each split file also carries `test_fixed`: a constant 20% holdout (subset of
every ratio's `test`). Use `test` for the Mohanty-style baseline comparison
(his protocol grows test as train shrinks) and `test_fixed` for the
training-size curve, where a moving test set would muddy the x-axis.

Committed split files: `color`/`grayscale` × ratios `0.8`/`0.5`/`0.2`, seed `42`.

## Running an experiment

Edit the values in `src/config.py` (architecture, variant, initialization,
ratio, batch size, epochs, device, etc.), then:

```bash
python -m src.main
```

This builds the model and data loaders, trains for `EPOCHS`, checkpoints the
best validation-F1 epoch, evaluates that checkpoint once on `test`, and
writes everything to `results/{architecture}_{variant}_{initialization}_{ratio}_{seed}/`:
`config.json`, `history.csv` (per-epoch metrics), `best_model.pt`, `summary.json`,
and `plots/` (accuracy curve, validation-F1 curve, confusion matrix).

A run refuses to overwrite an existing result directory unless
`OVERWRITE = True` in `config.py` -- delete the stale directory or flip that
flag if you're intentionally rerunning the same config.

`DEVICE = "auto"` picks CUDA > MPS > CPU. `AMP = True` only actually enables
mixed precision on CUDA (`device.type == "cuda"`) -- Apple Silicon has no
fp16 tensor cores, and `GradScaler`'s per-step inf/nan check forces a
device-host sync that made MPS training *slower* than plain CPU during
testing, so AMP is deliberately not applied there.

If you're on a memory-constrained machine (8GB unified memory or less) and
training looks stalled rather than slow -- check `sysctl vm.swapusage` before
assuming it's a code bug. `NUM_WORKERS`, `persistent_workers`, and
`prefetch_factor` in `src/data.py`'s `_loader()` are tuned down from more
common defaults for exactly this reason; lower `NUM_WORKERS` further or drop
`BATCH_SIZE` if it's still swapping.

## Loading data in code (`src/data.py`)

```python
from src.data import get_dataloaders

train_loader, test_loader = get_dataloaders(variant="color", ratio=0.8, seed=42, batch_size=64)
# or, with a validation split (requires val_frac > 0 when the split was generated):
train_loader, val_loader, test_loader = get_dataloaders(
    "color", 0.8, 42, batch_size=64, return_val=True
)
```

Yields `(image: Tensor[B,3,224,224], label: Tensor[B])`, labels `0..29` in the
order fixed by `data/classes.json`. Color and grayscale share one transform
pipeline (grayscale is replicated to 3 channels) so a color-vs-grayscale
comparison isn't confounded by different plumbing.

`get_dataloaders`'s required positional signature (`variant, ratio, seed,
batch_size`) is frozen in `src/interfaces.py`, alongside `build_model` and
`compute_metrics`. `src/data.py` extends it with backward-compatible
keyword-only args (`return_val`, `test_split`, etc.) without breaking the
frozen contract. `src/interfaces.py` requires all-team agreement to change.

## Metrics and plots (`src/metrics/`)

`f1_from_confusion()` derives macro and support-weighted F1 directly from an
accumulated confusion matrix (no separate per-batch bookkeeping needed).
`plot_metrics()` writes three PNGs per run into that run's `plots/`: train
vs. validation accuracy, validation F1 over epochs, and the confusion matrix
heatmap.

## Grad-CAM (`scripts/grad_cam.py`, `scripts/grad_cam_sample.py`)

Visualizes which regions of an image a trained checkpoint's prediction was
most sensitive to -- used to check whether a model is attending to actual
lesion/leaf features or to background/lighting artifacts. Both scripts read
their architecture, normalization, and class list straight from the
checkpoint's own saved `config`/`class_names`, so nothing needs to be passed
in beyond the checkpoint path and images.

**Specific images:**
```bash
python -m scripts.grad_cam --checkpoint results/<run_name>/best_model.pt path/to/image1.JPG path/to/image2.JPG
```

**Random sample of correct + incorrect test predictions:**
```bash
python -m scripts.grad_cam_sample --checkpoint results/<run_name>/best_model.pt --n-correct 5 --n-incorrect 5 --seed 0
```
This scores the checkpoint's entire test set first (one full inference pass,
takes a bit), then runs Grad-CAM only on the sampled images. `--seed` controls
which images get picked; the same seed reproduces the same sample.

Both default to writing under `results/<run_name>/gradcam/`, alongside that
run's `config.json`, `history.csv`, and `plots/` -- override with `--out` if
you want a different location. `grad_cam_sample.py` additionally splits into
`gradcam/correct/` and `gradcam/incorrect/`, each holding `_original` and
`_gradcam` pairs per image sharing a filename stem.

Target-layer resolution currently supports `mobilenet_v2` and `vanilla_cnn`
(this project's only two architectures) -- there's no ResNet support since
ResNet isn't a model option in `src/models` at all yet.

## Known limitation: leaf-map coverage

Beyond the 8 classes dropped entirely (see above), `leaf-map.json` also has
only partial coverage for `Tomato___Late_blight`, `Tomato___Septoria_leaf_spot`,
`Tomato___Tomato_Yellow_Leaf_Curl_Virus`, `Tomato___healthy`, and
`Strawberry___Leaf_scorch` (30-52% verified) -- these mix an `RS_*`-named
(leaf-tracked) batch with a `GHLB*`-named (greenhouse) batch that was never
leaf-tracked upstream.

Unmatched images within a surviving class are represented as singleton groups
in the manifest, but a singleton cannot prove it's a distinct physical leaf.
The default split generator excludes them, yielding a smaller but leak-safer
evaluation set for that class. Pass `--include-unverified-leaves` to
`make_splits.py` only when retaining all images matters more than that
guarantee; report the resulting leakage risk as a limitation.
`prepare_data.py` prints a per-class breakdown of any class under 50%
verified coverage on every run so this stays visible.

## License

Apache 2.0 (see `LICENSE`).
