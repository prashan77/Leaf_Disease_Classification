#!/usr/bin/env python
"""
prepare_data.py -- one-time dataset preparation.

Assumes data.zip and leaf_grouping/leaf-map.json are already sitting under
--root (e.g. because you downloaded them with `hf download` yourself). This
script does NOT hit the network. If either input is missing, it tells you
what to fetch and exits.

Outputs (small text files, safe to commit):
    <root>/manifest.csv        one row per image, per variant, with leaf_id
    <root>/class_counts.csv    class-count table (GM3 needs this for macro-F1)
    <root>/classes.json        frozen label ordering -> int index

Images themselves land at <root>/raw/{variant}/{class}/*.JPG and should be
gitignored.

Usage:
    python -m scripts.prepare_data --root data
    python -m scripts.prepare_data --root data --variants color grayscale segmented
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

EXPECTED_CLASSES = 38
EXPECTED_IMAGES = 54_305  # confirmed against HF's 43,596 train + 10,709 test


# --------------------------------------------------------------------------
# locate inputs (no downloading)
# --------------------------------------------------------------------------
def find_inputs(root: Path) -> tuple[Path, Path]:
    zip_path = root / "data.zip"
    leaf_map_path = root / "leaf_grouping" / "leaf-map.json"

    missing = []
    if not zip_path.exists():
        missing.append(f"  {zip_path}")
    if not leaf_map_path.exists():
        missing.append(f"  {leaf_map_path}")

    if missing:
        print("Missing input(s):")
        print("\n".join(missing))
        print("\nFetch with:")
        print("  hf download mohanty/PlantVillage data.zip leaf_grouping/leaf-map.json \\")
        print(f"    --repo-type dataset --local-dir {root}")
        sys.exit(1)

    return zip_path, leaf_map_path


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def _member_target(name: str, variants: set[str]) -> tuple[str, str, str] | None:
    """Map a zip member path to (variant, class_name, filename), or None to skip."""
    if name.endswith("/"):
        return None
    parts = name.split("/")
    if "raw" not in parts:
        return None
    i = parts.index("raw")
    if len(parts) < i + 4:
        return None
    variant, class_name, filename = parts[i + 1], parts[i + 2], parts[i + 3]
    if variant not in variants:
        return None
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return None
    return variant, class_name, filename


def extract(zip_path: Path, root: Path, variants: set[str]) -> None:
    print(f"[1/4] extracting {sorted(variants)} to {root}/raw/")
    written = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        targets = [(m, _member_target(m, variants)) for m in zf.namelist()]
        targets = [(m, t) for m, t in targets if t is not None]
        if not targets:
            sys.exit(f"No matching members found in {zip_path}. Check its contents with "
                      f"`python -c \"import zipfile;print(zipfile.ZipFile('{zip_path}').namelist()[:20])\"`")
        for member, (variant, class_name, filename) in targets:
            dest = root / "raw" / variant / class_name / filename
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())
            written += 1
            if written % 5000 == 0:
                print(f"      {written} written...")
    print(f"[1/4] done: {written} written, {skipped} already present")


# --------------------------------------------------------------------------
# leaf grouping
# --------------------------------------------------------------------------
def leaf_key(filename: str) -> str:
    """
    '0a37f34f-...___Mary_HL 9155.JPG'  ->  'mary_hl 9155'
    The leaf map is keyed on the lowercased source-photo token after the UUID.
    """
    stem = Path(filename).stem
    if "___" in stem:
        stem = stem.rsplit("___", 1)[1]
    return stem.strip().lower()


def resolve_leaf_id(key: str, class_name: str, leaf_map: dict) -> tuple[str, bool]:
    """
    A key can map to several classes (same photo token reused across e.g.
    Soybean___healthy and Apple___healthy), so disambiguate by directory when
    possible. Unmatched/ambiguous images become singleton groups -- safe: this
    can only split leaves apart, never merge distinct ones, so it cannot
    introduce leakage.

    One wrinkle discovered against the real map: some keys carry an older or
    differently-spelled class label than the current manifest (e.g. the map's
    "Apple_Frogeye Spot" vs. this dataset's "Apple___Black_rot", after an
    apparent taxonomy merge upstream). When a key has exactly one entry, the
    match is unambiguous regardless of label spelling, so we trust it.
    """
    entries = leaf_map.get(key, ())
    if not entries:
        return f"{class_name}:::solo_{key}", False

    # exact class match first -- handles keys that legitimately span
    # multiple real classes
    for entry in entries:
        cls, _, group = entry.rpartition(":::")
        if cls == class_name:
            return f"{class_name}:::{group}", True

    # unambiguous: exactly one leaf group under this key, just filed under a
    # different class string. Same physical leaf either way.
    if len(entries) == 1:
        cls, _, group = entries[0].rpartition(":::")
        return f"{cls}:::{group}", True

    # multiple different classes under this key, none matching ours ->
    # genuinely ambiguous, stay conservative
    return f"{class_name}:::solo_{key}", False


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def build_manifest(root: Path, variants: list[str], leaf_map: dict) -> None:
    print("[2/4] building manifest with leaf grouping")

    reference_variant = variants[0]
    reference_dir = root / "raw" / reference_variant
    classes = sorted(p.name for p in reference_dir.iterdir() if p.is_dir())
    if len(classes) != EXPECTED_CLASSES:
        print(f"  !! found {len(classes)} classes, expected {EXPECTED_CLASSES}")

    expected_classes = set(classes)
    for variant in variants[1:]:
        found_classes = {p.name for p in (root / "raw" / variant).iterdir() if p.is_dir()}
        if found_classes != expected_classes:
            missing = sorted(expected_classes - found_classes)
            extra = sorted(found_classes - expected_classes)
            raise ValueError(
                f"Class directories for {variant} differ from {reference_variant}; "
                f"missing={missing}, extra={extra}"
            )

    label_of = {c: i for i, c in enumerate(classes)}
    (root / "classes.json").write_text(json.dumps(classes, indent=2))

    rows = []
    matched = unmatched = 0
    per_variant = Counter()

    for variant in variants:
        vdir = root / "raw" / variant
        for class_name in classes:
            cdir = vdir / class_name
            if not cdir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {variant}/{class_name}")
            for img in sorted(cdir.iterdir()):
                if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                key = leaf_key(img.name)
                lid, ok = resolve_leaf_id(key, class_name, leaf_map)
                matched += ok
                unmatched += not ok
                per_variant[variant] += 1
                rows.append({
                    "relpath": f"{class_name}/{img.name}",
                    "variant": variant,
                    "class_name": class_name,
                    "label": label_of[class_name],
                    "leaf_id": lid,
                })

    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["relpath", "variant", "class_name", "label", "leaf_id"])
        w.writeheader()
        w.writerows(rows)

    total = matched + unmatched
    print(f"  leaf-map coverage: {matched}/{total} ({100 * matched / max(total, 1):.1f}%)")
    if unmatched:
        print(
            f"  {unmatched} images lack verified leaf groups. They are represented as "
            "singletons and are excluded from default splits to avoid possible leakage."
        )
        _warn_uncovered_classes(rows, variants[0])
    for v, n in per_variant.items():
        flag = "" if n == EXPECTED_IMAGES else f"  <-- expected {EXPECTED_IMAGES}"
        print(f"  {v}: {n} images{flag}")

    _write_class_counts(root, rows, variants[0], classes)


def _warn_uncovered_classes(rows: list[dict], variant: str) -> None:
    """
    Flag classes where leaf-map.json coverage is low/zero. Some classes (or
    photography batches within a class) were never leaf-tracked upstream, so
    there is no leaf identity to recover. Such images are excluded from the
    default leak-safe split, rather than being treated as safe singleton groups.
    """
    per_class = Counter()
    per_class_matched = Counter()
    for r in rows:
        if r["variant"] != variant:
            continue
        per_class[r["class_name"]] += 1
        per_class_matched[r["class_name"]] += ":::solo_" not in r["leaf_id"]

    flagged = [
        (c, per_class_matched[c], n)
        for c, n in per_class.items()
        if per_class_matched[c] / n < 0.5
    ]
    if not flagged:
        return
    print("  low/zero verified leaf-map coverage by class:")
    for c, matched, n in sorted(flagged, key=lambda t: t[1] / t[2]):
        print(f"    {c}: {matched}/{n} ({100 * matched / n:.0f}%)")


def _write_class_counts(root: Path, rows: list[dict], variant: str, classes: list[str]) -> None:
    print("[3/4] writing class-count table")
    img_counts = Counter()
    leaf_sets = defaultdict(set)
    for r in rows:
        if r["variant"] != variant:
            continue
        img_counts[r["class_name"]] += 1
        leaf_sets[r["class_name"]].add(r["leaf_id"])

    with open(root / "class_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "class_name", "n_images", "n_leaves", "imgs_per_leaf"])
        for c in classes:
            n_img, n_leaf = img_counts[c], len(leaf_sets[c])
            w.writerow([classes.index(c), c, n_img, n_leaf, f"{n_img / max(n_leaf, 1):.1f}"])

    ordered = sorted(img_counts.items(), key=lambda kv: kv[1])
    print(f"  rarest:  {ordered[0][0]} ({ordered[0][1]} images, {len(leaf_sets[ordered[0][0]])} leaves)")
    print(f"  largest: {ordered[-1][0]} ({ordered[-1][1]} images, {len(leaf_sets[ordered[-1][0]])} leaves)")
    print(f"  imbalance ratio: {ordered[-1][1] / max(ordered[0][1], 1):.0f}x  -- GM3 needs this for macro-F1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--variants", nargs="+", default=["color", "grayscale"],
                    choices=["color", "grayscale", "segmented"])
    args = ap.parse_args()

    zip_path, leaf_map_path = find_inputs(args.root)
    extract(zip_path, args.root, set(args.variants))
    leaf_map = json.loads(leaf_map_path.read_text())
    build_manifest(args.root, args.variants, leaf_map)

    digest = hashlib.sha256((args.root / "manifest.csv").read_bytes()).hexdigest()[:16]
    print(f"[4/4] manifest.csv sha256[:16] = {digest}")
    print("      All four teammates must see this same digest. Post it in the channel.")


if __name__ == "__main__":
    main()
