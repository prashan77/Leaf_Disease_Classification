# Leaf_Disease_Classification

CS7643 final project: leaf disease classification on the [PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage)
dataset (38 classes, ~54k images per variant), with a leaf-grouped, class-stratified
split protocol so augmented copies of the same physical leaf never straddle
train/test.

## Repo layout

```
scripts/
  prepare_data.py   one-time: zip + leaf-map -> data/raw/, manifest.csv, classes.json
  make_splits.py    manifest.csv -> splits/*.json (train/val/test index files)
src/
  data.py           PlantVillageDataset + get_dataloaders() used by training code
data/
  manifest.csv        committed -- one row per image per variant, with leaf_id
  classes.json         committed -- frozen label ordering -> int index
  class_counts.csv     committed -- per-class image/leaf counts, for macro-F1 weighting
  data.zip, raw/, leaf_grouping/   gitignored -- regenerate locally, see below
splits/
  {variant}_{ratio}_{seed}.json   committed -- e.g. color_0.8_42.json
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pillow huggingface_hub matplotlib
```

(No `requirements.txt` yet -- these are the packages actually imported by
`scripts/` and `src/` as of this commit: `torch` 2.13, `torchvision` 0.28,
`numpy` 2.5, `Pillow` 12.3, `huggingface_hub` 1.24, `matplotlib`.)

## Rebuilding the data pipeline

The raw images and leaf map are gitignored (too large / not ours to redistribute),
so anyone starting fresh needs to regenerate `data/raw/`, then the splits:

```bash
# 1. fetch the zip + leaf map (network, one time)
hf download mohanty/PlantVillage data.zip leaf_grouping/leaf-map.json \
    --repo-type dataset --local-dir data

# 2. extract + build manifest.csv / classes.json / class_counts.csv (offline)
python -m scripts.prepare_data --root data

# 3. build the leaf-grouped, class-stratified, nested splits (offline)
python -m scripts.make_splits --root data --ratios 0.8 0.5 0.2 --seed 42
```

Both scripts print a sha256 digest of what they produce
(`manifest.csv` in step 2, each `splits/*.json` in step 3). Everyone on the
team should get identical digests -- if yours differs, something upstream
(zip version, leaf map, seed) doesn't match.

`prepare_data.py --variants` defaults to `color grayscale`; pass
`--variants color grayscale segmented` if you also need the segmented variant
(present in the zip, not extracted by default).

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

## Loading data in training code (`src/data.py`)

```python
from src.data import get_dataloaders

train_loader, test_loader = get_dataloaders(variant="color", ratio=0.8, seed=42, batch_size=64)
# or, with a validation split (requires val_frac > 0 when the split was generated):
train_loader, val_loader, test_loader = get_dataloaders(
    "color", 0.8, 42, batch_size=64, return_val=True
)
```

Yields `(image: Tensor[B,3,224,224], label: Tensor[B])`, labels `0..37` in the
order fixed by `data/classes.json`. Color and grayscale share one transform
pipeline (grayscale is replicated to 3 channels) so a color-vs-grayscale
comparison isn't confounded by different plumbing.

`get_dataloaders`'s required positional signature (`variant, ratio, seed,
batch_size`) is frozen in `src/interfaces.py`, alongside `build_model` and
`compute_metrics` for GM2/GM3. `src/data.py` extends it with backward-compatible
keyword-only args (`return_val`, `test_split`, etc.) without breaking the frozen
contract.

## Known limitation: leaf-map coverage

`leaf-map.json` only resolves ~76% of images to a verified physical-leaf
group; coverage is not evenly spread -- it's **zero** for:

- `Corn_(maize)___*` (all 4 classes)
- `Grape___healthy`
- `Squash___Powdery_mildew`
- `Tomato___Target_Spot`
- `Tomato___Tomato_mosaic_virus`

and partial (30-52%) for `Tomato___Late_blight`, `Tomato___Septoria_leaf_spot`,
`Tomato___Tomato_Yellow_Leaf_Curl_Virus`, `Tomato___healthy`, and
`Strawberry___Leaf_scorch` -- these mix an `RS_*`-named (leaf-tracked) batch
with a `GHLB*`-named (greenhouse) batch that was never leaf-tracked upstream.

Unmatched images are represented as singleton groups in the manifest, but a
singleton cannot prove that it is a distinct physical leaf. The default split
generator excludes them, yielding a smaller but leak-safer evaluation set.
Pass `--include-unverified-leaves` only when retaining all images matters more
than that guarantee; report the resulting leakage risk as a limitation.
`prepare_data.py` prints a per-class breakdown of any class under 50% coverage
on every run so this stays visible.

## License

Apache 2.0 (see `LICENSE`).
