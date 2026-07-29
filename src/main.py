"""Start a training run using the values in src/config.py."""

from __future__ import annotations

from src.training.experiment import run_experiment


def main() -> None:
    """Run the configured experiment and show readable configuration errors."""

    try:
        run_experiment()
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
