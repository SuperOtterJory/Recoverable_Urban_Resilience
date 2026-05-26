"""Run the recoverable-resilience data-mining pipeline."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_mining.yml")
    args = parser.parse_args()
    raise SystemExit(
        "Data-mining pipeline is scaffolded. Implementation will be added in the next version. "
        f"Config: {args.config}"
    )


if __name__ == "__main__":
    main()
