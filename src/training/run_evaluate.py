from pathlib import Path

import torch

from src.training.evaluate import build_loaders, build_model_from_config, collect_predictions, evaluate_model, save_predictions
from src.training.train import load_config


def evaluate_checkpoint(config_path: str | Path, checkpoint_path: str | Path, predictions_csv: str | Path | None = None) -> dict:
    """Load a checkpoint and evaluate it on the configured test split."""
    config = load_config(config_path)
    device = torch.device(config.get("training", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    _train_loader, _val_loader, test_loader = build_loaders(config)
    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, test_loader, device, high_tc_threshold=float(config.get("evaluation", {}).get("high_tc_threshold", 77.0)))

    output_path = predictions_csv or config.get("evaluation", {}).get("predictions_csv")
    if output_path:
        preds = collect_predictions(model, test_loader, device)
        save_predictions(preds, output_path)
    return metrics
