"""Multi-seed ablation for robust evaluation (Phase 3).

Runs each (config × seed) pair sequentially on the single available GPU,
then aggregates test metrics into mean ± std across seeds.

Usage:
    python scripts/run_multiseed_ablation.py [--seeds 0 1 2 3 4] [--output-dir results/multiseed]
"""

import argparse
import copy
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from src.training.train import load_config, train_from_config


DEFAULT_CONFIGS = [
    ("dft", "configs/ablation/dft.yaml"),
    ("crys_jepa_dft", "configs/ablation/crys_jepa_dft.yaml"),
]

METRIC_KEYS = [
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "mae", "rmse", "mae_superconductors", "mae_high_tc",
]


def _flatten(mode: str, seed: int, result: dict) -> dict:
    """Flatten one seed run into a stable per-seed CSV row."""
    cls = result["test_metrics"]["classification"]
    reg = result["test_metrics"]["regression"]
    return {
        "input_mode": mode,
        "seed": seed,
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


def _aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate per-seed rows into mean/std summaries by input mode."""
    by_mode: dict[str, list] = defaultdict(list)
    for row in rows:
        by_mode[row["input_mode"]].append(row)

    summary = []
    for mode, mode_rows in by_mode.items():
        agg: dict = {"input_mode": mode, "n_seeds": len(mode_rows)}
        for key in METRIC_KEYS:
            vals = [r[key] for r in mode_rows if r.get(key) is not None]
            agg[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            agg[f"{key}_std"] = float(np.std(vals)) if vals else None
        summary.append(agg)
    return summary


def _save_csv(rows: list[dict], path: Path) -> None:
    """Write rows to CSV, creating the parent directory first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(seeds: list[int], configs: list[tuple[str, str]], output_dir: str) -> None:
    """Run every config/seed pair and save detailed plus aggregate metrics."""
    per_seed_rows: list[dict] = []
    total = len(configs) * len(seeds)
    done = 0

    for mode, config_path in configs:
        base_config = load_config(config_path)
        for seed in seeds:
            done += 1
            print(f"\n[{done}/{total}] === {mode} | seed={seed} ===")
            config = copy.deepcopy(base_config)
            config["training"]["seed"] = seed
            config["checkpoints"]["save_dir"] = f"checkpoints/multiseed/{mode}/seed_{seed}"
            result = train_from_config(config)
            row = _flatten(mode, seed, result)
            per_seed_rows.append(row)
            print(f"  → test_f1={row['f1']:.4f}  test_mae={row['mae']:.4f}  test_roc_auc={row['roc_auc']:.4f}")

    output_path = Path(output_dir)
    per_seed_csv = output_path / "multiseed_per_seed.csv"
    _save_csv(per_seed_rows, per_seed_csv)
    print(f"\nRésultats par seed → {per_seed_csv}")

    summary = _aggregate(per_seed_rows)
    summary_csv = output_path / "multiseed_summary.csv"
    _save_csv(summary, summary_csv)
    print(f"Résumé (moyenne±std) → {summary_csv}")

    print("\n=== Rapport de stabilité ===")
    for row in summary:
        m = row["input_mode"]
        f1 = row["f1_mean"]
        f1s = row["f1_std"]
        mae = row["mae_mean"]
        maes = row["mae_std"]
        roc = row["roc_auc_mean"]
        rocs = row["roc_auc_std"]
        print(f"  {m:20s}  F1={f1:.4f}±{f1s:.4f}  MAE={mae:.4f}±{maes:.4f}  ROC-AUC={roc:.4f}±{rocs:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed ablation for robust evaluation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)),
                        help="Seeds to run (default: 0 1 2 3 4).")
    parser.add_argument("--output-dir", default="results/multiseed",
                        help="Output directory for CSVs.")
    args = parser.parse_args()
    main(args.seeds, DEFAULT_CONFIGS, args.output_dir)
