"""Run the supervised input-mode ablation and write a metrics CSV."""

import argparse
import csv
from pathlib import Path

from src.training.train import load_config, train_from_config


DEFAULT_CONFIGS = [
    ("crys_jepa", "configs/ablation/crys_jepa.yaml"),
    ("dft", "configs/ablation/dft.yaml"),
    ("crys_jepa_dft", "configs/ablation/crys_jepa_dft.yaml"),
]


def _flatten_result(mode: str, result: dict) -> dict:
    """Flatten nested training metrics into one CSV row."""
    cls = result["test_metrics"]["classification"]
    reg = result["test_metrics"]["regression"]
    return {
        "input_mode": mode,
        "best_checkpoint": result["best_checkpoint"],
        "best_val_mae": result["best_val_mae"],
        "accuracy": cls.get("accuracy"),
        "precision": cls.get("precision"),
        "recall": cls.get("recall"),
        "f1": cls.get("f1"),
        "roc_auc": cls.get("roc_auc"),
        "mae": reg.get("mae"),
        "rmse": reg.get("rmse"),
        "mae_superconductors": reg.get("mae_superconductors"),
        "mae_high_tc": reg.get("mae_high_tc"),
    }


def main(output_csv: str) -> None:
    """Train each default ablation config and persist the comparison table."""
    rows = []
    for mode, config_path in DEFAULT_CONFIGS:
        print(f"\n=== Training {mode} ===")
        config = load_config(config_path)
        result = train_from_config(config)
        rows.append(_flatten_result(mode, result))

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ablation results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crys-JEPA/DFT ablations and save a metrics table.")
    parser.add_argument("--output", default="ablation_results.csv", help="Output CSV path.")
    args = parser.parse_args()
    main(args.output)
