from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

DATASET_REPO = "mohanty/PlantVillage"
REQUIRED_FILES = (
    "data.zip",
    "leaf_grouping/leaf-map.json",
)


def _download_from_hf(filename: str, root: Path, force: bool) -> Path:
    try:
        source = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=root,
            force_download=force,
        )
    except Exception as exc:
        raise SystemExit("could not download dataset from hf")
    
    return Path(source)


def download_data(root: Path, force: bool = False) -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []

    for filename in REQUIRED_FILES:
        destination = root / filename
        if destination.is_file() and destination.stat().st_size > 0 and not force:
            print(f"[skip] {destination} already exists")
            downloaded_paths.append(destination)
            continue

        print(f"[download] {DATASET_REPO}/{filename}")
        path = _download_from_hf(filename, root, force)


        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Download did not produce a valid file: {path}")
        
        print(f"[done] {path} ({path.stat().st_size / (1024 * 1024):.1f} MB)")
        downloaded_paths.append(path)

    print("\nPlantVillage inputs are ready.")
    return downloaded_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PlantVillage inputs required by prepare_data.py."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="destination directory (default: data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload files even when non-empty local copies exist",
    )
    args = parser.parse_args()
    download_data(args.root, args.force)


if __name__ == "__main__":
    main()
