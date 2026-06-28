"""CLI wrapper for single-CIF superconductivity inference."""

import argparse

import _bootstrap  # noqa: F401
from src.infer import format_prediction, predict_cif


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict superconductivity from one CIF.")
    parser.add_argument("--checkpoint", required=True, help="Path to supervised checkpoint.")
    parser.add_argument("--cif", required=True, help="Path to a CIF file.")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda.")
    args = parser.parse_args()
    print(format_prediction(predict_cif(args.checkpoint, args.cif, device=args.device)))
