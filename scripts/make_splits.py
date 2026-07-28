#!/usr/bin/env python
"""
make_splits.py -- generate and commit the train/val/test index files.

Three properties, each chosen deliberately:

  1. LEAF-GROUPED. All photographs of one physical leaf land on the same side of
     the split. PlantVillage shoots each leaf several times; a random split puts
     near-duplicates in the test set and inflates every number you report. The
     HF dataset card is explicit that custom ratios must respect leaf_id.

  2. CLASS-STRATIFIED. Leaves are shuffled and cut within each class, not
     globally. The dataset spans ~150 to ~5500 images per class (36x imbalance);
     a global cut at ratio 0.2 can starve a rare class of training data or,
     worse, of test data, which silently corrupts macro-F1.

  3. NESTED. One shuffle per class, reused across ratios. train(0.2) is a strict
     subset of train(0.5) is a strict subset of train(0.8). The accuracy-vs-
     training-set-size curve then varies exactly one thing -- how much data --
     instead of confounding size with which particular leaves got drawn.

Each split file also carries `test_fixed`: the 20% holdout from the 80-20 split,
which is a subset of every ratio's test set. Evaluate on `test` for the
Mohanty comparison table (his protocol grows the test set as train shrinks), and
on `test_fixed` for the headline curve, where a moving test set would muddy it.
Both come free -- they are index lists, not extra training runs.

Note on leaf_id: ~75.7% of images resolve to leaf-map-verified groups; the rest
(concentrated in Corn and Squash classes, absent from the source leaf map)
fall back to per-image singleton groups. That is conservative, not leaky --
it can only split a leaf's photos apart, never merge two different leaves --
so grouping here is exactly as strict as the manifest provides.

Usage:
    python -m scripts.make_splits --root data --ratios 0.8 0.5 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_manifest(root: Path, variant: str) -> list[dict]:
    with open(root / "manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == variant]
    if not rows:
        raise SystemExit(f"No rows for variant={variant}. Run scripts/prepare_data.py first.")
    return rows


def has_verified_leaf_id(row: dict) -> bool:
    """Only these groups can support the no-cross-leaf guarantee."""
    return ":::solo_" not in row["leaf_id"]


def shuffled_leaves_per_class(rows: list[dict], seed: int) -> dict[str, list[str]]:
    """One deterministic shuffle per class. Reused by every ratio -> nesting."""
    by_class: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_class[r["class_name"]].add(r["leaf_id"])

    rng = np.random.default_rng(seed)
    out = {}
    for cls in sorted(by_class):                 # sorted -> order independent of dict insertion
        leaves = np.array(sorted(by_class[cls]))  # sorted -> order independent of filesystem
        rng.shuffle(leaves)
        out[cls] = leaves.tolist()
    return out


def cut(n: int, ratio: float) -> int:
    """Index of the train/test boundary, leaving at least one leaf on each side."""
    return max(1, min(n - 1, int(round(ratio * n))))


def build_split(
    rows: list[dict],
    leaves_by_class: dict[str, list[str]],
    ratio: float,
    val_frac: float,
    holdout_ratio: float,
) -> dict[str, list[str]]:
    paths_by_leaf: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        paths_by_leaf[r["leaf_id"]].append(r["relpath"])

    train, val, test, test_fixed = [], [], [], []

    for cls, leaves in leaves_by_class.items():
        n = len(leaves)
        k = cut(n, ratio)
        tr_leaves, te_leaves = leaves[:k], leaves[k:]

        # the same 20% holdout for every ratio, so the curve has a stable x-axis
        fixed_leaves = leaves[cut(n, holdout_ratio):]

        n_val = int(round(val_frac * len(tr_leaves))) if val_frac > 0 else 0
        n_val = min(n_val, len(tr_leaves) - 1) if len(tr_leaves) > 1 else 0
        if n_val:
            val_leaves = tr_leaves[-n_val:]
            tr_leaves = tr_leaves[:-n_val]
        else:
            val_leaves = []

        for bucket, leaf_ids in (
            (train, tr_leaves), (val, val_leaves), (test, te_leaves), (test_fixed, fixed_leaves)
        ):
            for lid in leaf_ids:
                bucket.extend(paths_by_leaf[lid])

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "test_fixed": sorted(test_fixed),
    }


def sanity_check(split: dict[str, list[str]], ratio: float) -> None:
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    assert not (tr & te), "train/test overlap"
    assert not (tr & va), "train/val overlap"
    assert not (va & te), "val/test overlap"
    assert set(split["test_fixed"]) <= te, "test_fixed must be a subset of test"
    total = len(tr) + len(va) + len(te)
    print(
        f"    ratio={ratio:g}  train={len(tr):>6}  val={len(va):>5}  "
        f"test={len(te):>6}  test_fixed={len(split['test_fixed']):>5}  "
        f"(train+val = {100 * (len(tr) + len(va)) / total:.1f}% of {total})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--variants", nargs="+", default=["color", "grayscale"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.8, 0.5, 0.2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="fraction of TRAIN leaves held out for model selection; "
                         "0 disables it (see warning)")
    ap.add_argument("--holdout-ratio", type=float, default=0.8,
                    help="ratio defining the fixed evaluation holdout")
    ap.add_argument(
        "--include-unverified-leaves",
        action="store_true",
        help="include singleton fallback groups; this weakens the leaf-leakage guarantee",
    )
    args = ap.parse_args()

    if not 0 <= args.val_frac < 1:
        ap.error("--val-frac must be in [0, 1).")
    if not 0 < args.holdout_ratio < 1:
        ap.error("--holdout-ratio must be in (0, 1).")
    invalid_ratios = [ratio for ratio in args.ratios if not 0 < ratio <= args.holdout_ratio]
    if invalid_ratios:
        ap.error(
            "Each --ratio must be in (0, --holdout-ratio] so test_fixed remains "
            f"a subset of test; got {invalid_ratios}."
        )

    if args.val_frac == 0:
        print("!! val_frac=0: train.py will have to checkpoint on the test set.")
        print("!! That is selection on test. If you do this, say so in Limitations.\n")

    args.out.mkdir(parents=True, exist_ok=True)
    reference: dict[float, list[str]] = {}

    for variant in args.variants:
        rows = load_manifest(args.root, variant)
        if not args.include_unverified_leaves:
            total = len(rows)
            rows = [row for row in rows if has_verified_leaf_id(row)]
            print(f"  {variant}: excluded {total - len(rows)} unverified-leaf images")
        elif any(not has_verified_leaf_id(row) for row in rows):
            print("  !! including unverified singleton groups; leakage cannot be ruled out")
        leaves_by_class = shuffled_leaves_per_class(rows, args.seed)
        n_leaves = sum(len(v) for v in leaves_by_class.values())
        print(f"  {variant}: {len(rows)} images across {n_leaves} physical leaves "
              f"({len(rows) / n_leaves:.1f} images per leaf)")

        for ratio in args.ratios:
            split = build_split(rows, leaves_by_class, ratio, args.val_frac, args.holdout_ratio)
            sanity_check(split, ratio)

            # Color and grayscale must resolve to identical partition paths,
            # otherwise the comparison is not controlled.
            if ratio in reference:
                assert split == reference[ratio], \
                    f"{variant} split differs from the first variant at ratio {ratio}"
            else:
                reference[ratio] = split

            payload = {
                "variant": variant,
                "ratio": ratio,
                "seed": args.seed,
                "grouping": "physical_leaf",
                "stratified_by": "class",
                "nested": True,
                "val_frac": args.val_frac,
                "includes_unverified_leaves": args.include_unverified_leaves,
                **split,
            }
            path = args.out / f"{variant}_{ratio:g}_{args.seed}.json"
            path.write_text(json.dumps(payload))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            print(f"      -> {path}  sha256[:16]={digest}")

    print("\nCommit splits/. Day 3 checksum check:  sha256sum splits/*.json")


if __name__ == "__main__":
    main()
