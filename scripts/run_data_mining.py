"""Run the recoverable-resilience data-mining pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from recoverable_resilience.data_mining import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_mining.yml")
    parser.add_argument(
        "--root",
        default=None,
        help="Repository root. Defaults to the current working directory/project root.",
    )
    args = parser.parse_args()
    run_pipeline(config_path=Path(args.config), root=Path(args.root) if args.root else None)


if __name__ == "__main__":
    main()
