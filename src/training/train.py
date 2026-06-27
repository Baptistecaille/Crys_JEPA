from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.evaluate import build_loaders, build_model_from_config, evaluate_model
from src.training.losses import SuperconductivityLoss
from src.utils.seed import set_seed


def load_config(path: str | Path) -> dict:
    """Load a YAML config file as a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, criterion: SuperconductivityLoss, optimizer, device) -> dict[str, float]:
    """Run one supervised training epoch."""
    model.train()
    totals = {"loss": 0.0, "loss_cls": 0.0, "loss_tc": 0.0}
    n_batches = 0
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, parts = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        for key in totals:
            totals[key] += parts[key]
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def _checkpoint_path(checkpoints: dict) -> Path:
    """Return the final checkpoint path and ensure its parent exists."""
    save_dir = Path(checkpoints.get("save_dir", "checkpoints")).expanduser()
    best_name = Path(checkpoints.get("best_name", "best.pt")).expanduser()
    best_path = best_name if best_name.is_absolute() else save_dir / best_name
    best_path.parent.mkdir(parents=True, exist_ok=True)
    return best_path


def _save_checkpoint(payload: dict, path: Path) -> None:
    """Save a checkpoint atomically enough to avoid leaving partial best files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Could not save checkpoint to {path.resolve()}") from exc


def train_from_config(config: dict) -> dict:
    """Train the MVP model and save the best validation checkpoint."""
    training = config["training"]
    checkpoints = config.get("checkpoints", {})
    set_seed(int(training.get("seed", 42)))
    device = torch.device(training.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_loaders(config)
    model = build_model_from_config(config).to(device)
    criterion = SuperconductivityLoss(
        lambda_tc=float(training.get("lambda_tc", 1.0)),
        regression_loss=training.get("regression_loss", "smooth_l1"),
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )

    best_path = _checkpoint_path(checkpoints)
    best_score = float("inf")
    history = []

    for epoch in tqdm(range(1, int(training.get("epochs", 50)) + 1), desc="Training 3DSC MVP"):
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device, high_tc_threshold=float(config.get("evaluation", {}).get("high_tc_threshold", 77.0)))
        val_mae = val_metrics["regression"]["mae"]
        record = {"epoch": epoch, "train": train_losses, "val": val_metrics}
        history.append(record)
        print(f"epoch={epoch} loss={train_losses['loss']:.4f} val_mae={val_mae:.4f} val_f1={val_metrics['classification']['f1']:.4f}")

        if val_mae < best_score:
            best_score = val_mae
            _save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device, high_tc_threshold=float(config.get("evaluation", {}).get("high_tc_threshold", 77.0)))
    return {"best_checkpoint": str(best_path), "best_val_mae": best_score, "test_metrics": test_metrics, "history": history}


def main(config_path: str | Path) -> dict:
    """CLI-friendly training entrypoint."""
    config = load_config(config_path)
    result = train_from_config(config)
    print(f"Best checkpoint: {result['best_checkpoint']}")
    print(f"Test metrics: {result['test_metrics']}")
    return result
