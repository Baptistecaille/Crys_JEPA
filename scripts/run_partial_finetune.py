import argparse
import csv
from pathlib import Path

from src.training.train import load_config, train_from_config


CONFIGS = [
    ("frozen", "configs/finetune_dft_jepa/frozen.yaml"),
    ("last1", "configs/finetune_dft_jepa/last1.yaml"),
    ("last2", "configs/finetune_dft_jepa/last2.yaml"),
]


def _flatten(mode: str, result: dict) -> dict:
    cls = result["test_metrics"]["classification"]
    reg = result["test_metrics"]["regression"]
    return {
        "mode": mode,
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


def main(checkpoint: str, matrix_scaler: str | None, output: str) -> None:
    rows = []
    for mode, config_path in CONFIGS:
        config = load_config(config_path)
        config["model"]["crys_jepa_checkpoint"] = checkpoint
        if matrix_scaler:
            config["model"]["crys_jepa_matrix_scaler"] = matrix_scaler
        print(f"\n=== Training {mode} with {checkpoint} ===")
        result = train_from_config(config)
        rows.append(_flatten(mode, result))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Fine-tuning comparison written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run frozen/partial fine-tuning Crys-JEPA + DFT experiments.")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained Crys-JEPA checkpoint.")
    parser.add_argument("--matrix-scaler", default="data/jepa/mean_std_scaler.pt", help="Path to Crys-JEPA matrix scaler.")
    parser.add_argument("--output", default="partial_finetune_results.csv", help="Output CSV path.")
    args = parser.parse_args()
    main(args.checkpoint, args.matrix_scaler, args.output)
