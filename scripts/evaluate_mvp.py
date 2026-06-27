import argparse

import _bootstrap  # noqa: F401
from src.training.run_evaluate import evaluate_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the 3DSC superconductivity MVP.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to supervised checkpoint.")
    parser.add_argument("--predictions-csv", default=None, help="Optional output CSV for predictions.")
    args = parser.parse_args()
    print(evaluate_checkpoint(args.config, args.checkpoint, predictions_csv=args.predictions_csv))
