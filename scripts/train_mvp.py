"""CLI wrapper for training the 3DSC superconductivity MVP."""

import argparse

import _bootstrap  # noqa: F401
from src.training.train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 3DSC superconductivity MVP.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    main(args.config)
