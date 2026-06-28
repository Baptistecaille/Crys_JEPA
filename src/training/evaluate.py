"""Evaluation and dataloader construction for supervised superconductivity models."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.threedsc_dataset import ThreeDSCDataset, collate_crystals, split_dataset
from src.models.crys_jepa_wrapper import build_crys_jepa_encoder
from src.models.superconductivity_heads import SuperconductivityCrysJEPA
from src.utils.dft_features import DFTFeatureScaler
from src.utils.metrics import classification_metrics, regression_metrics


def move_batch_to_device(batch: dict, device: torch.device | str) -> dict:
    """Move tensor values of a collated crystal batch to a device."""
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device | str) -> dict:
    """Run inference over a loader and collect tensors plus metadata."""
    model.eval()
    outputs = {"tc_true": [], "tc_pred": [], "label_true": [], "logit_supra": [], "formula": [], "cif_path": []}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred = model(batch)
            outputs["tc_true"].append(batch["Tc"].detach().cpu())
            outputs["tc_pred"].append(pred["tc"].detach().cpu())
            outputs["label_true"].append(batch["label_supra"].detach().cpu())
            outputs["logit_supra"].append(pred["logit_supra"].detach().cpu())
            outputs["formula"].extend(batch["formula"])
            outputs["cif_path"].extend(batch["cif_path"])

    for key in ("tc_true", "tc_pred", "label_true", "logit_supra"):
        outputs[key] = torch.cat(outputs[key], dim=0)
    return outputs


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device | str, high_tc_threshold: float = 77.0) -> dict:
    """Evaluate classification and Tc regression in one pass."""
    preds = collect_predictions(model, loader, device)
    return {
        "classification": classification_metrics(preds["logit_supra"], preds["label_true"]),
        "regression": regression_metrics(preds["tc_pred"], preds["tc_true"], high_tc_threshold=high_tc_threshold),
    }


def save_predictions(preds: dict, output_path: str | Path) -> None:
    """Save per-material predictions as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "formula": preds["formula"],
            "cif_path": preds["cif_path"],
            "Tc_true": preds["tc_true"].tolist(),
            "Tc_pred": preds["tc_pred"].tolist(),
            "label_true": preds["label_true"].tolist(),
            "prob_supra": torch.sigmoid(preds["logit_supra"]).tolist(),
        }
    )
    frame.to_csv(output_path, index=False)


def build_model_from_config(config: dict) -> SuperconductivityCrysJEPA:
    """Construct the supervised model from a plain config dictionary."""
    encoder = build_crys_jepa_encoder(config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    dft_columns = config.get("data", {}).get("dft_features", [])
    return SuperconductivityCrysJEPA(
        crys_jepa_encoder=encoder,
        z_dim=int(model_cfg.get("z_dim", getattr(encoder, "z_dim", 512))),
        hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
        freeze_encoder=bool(train_cfg.get("freeze_encoder", True)),
        use_uncertainty=bool(model_cfg.get("use_uncertainty", False)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        input_mode=model_cfg.get("input_mode", "crys_jepa"),
        dft_feature_dim=int(model_cfg.get("dft_feature_dim", len(dft_columns))),
        dft_embedding_dim=int(model_cfg.get("dft_embedding_dim", 64)),
    )


def build_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders from config."""
    data_cfg = config["data"]
    split_cfg = config["splits"]
    train_cfg = config["training"]
    dft_columns = data_cfg.get("dft_features", []) if config.get("model", {}).get("input_mode", "crys_jepa") in {"dft", "crys_jepa_dft"} else []
    dataset = ThreeDSCDataset(
        csv_path=data_cfg["csv_path"],
        cif_dir=data_cfg["cif_dir"],
        formula_column=data_cfg.get("formula_column", "formula"),
        tc_column=data_cfg.get("tc_column", "Tc"),
        cif_column=data_cfg.get("cif_column", "cif_path"),
        csv_comment=data_cfg.get("csv_comment"),
        dft_feature_columns=dft_columns,
        primitive=bool(data_cfg.get("primitive", False)),
        reduced=bool(data_cfg.get("reduced", False)),
    )
    train_set, val_set, test_set = split_dataset(
        dataset,
        train=float(split_cfg.get("train", 0.8)),
        val=float(split_cfg.get("val", 0.1)),
        test=float(split_cfg.get("test", 0.1)),
        seed=int(train_cfg.get("seed", 42)),
    )
    if dft_columns:
        train_values = dataset.raw_dft_matrix()[train_set.indices]
        scaler_state = config.get("runtime", {}).get("dft_scaler")
        scaler = DFTFeatureScaler.from_state_dict(scaler_state) if scaler_state else DFTFeatureScaler.fit(train_values)
        dataset.set_dft_scaler(scaler)
        config.setdefault("runtime", {})["dft_scaler"] = scaler.state_dict()
        config.setdefault("model", {})["dft_feature_dim"] = len(dft_columns)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 2))
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent = num_workers > 0
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_crystals,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent,
        "prefetch_factor": 2 if persistent else None,
    }
    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
        DataLoader(test_set, shuffle=False, **loader_kwargs),
    )
