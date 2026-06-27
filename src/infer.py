from pathlib import Path

import torch
import yaml

from src.datasets.threedsc_dataset import collate_crystals
from src.models.crys_jepa_wrapper import build_crys_jepa_encoder
from src.models.superconductivity_heads import SuperconductivityCrysJEPA
from src.utils.cif_utils import load_cif_tensors


def load_model_for_inference(checkpoint_path: str | Path, device: str | torch.device | None = None) -> tuple[SuperconductivityCrysJEPA, dict, torch.device]:
    """Load a supervised checkpoint and rebuild its model."""
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    encoder = build_crys_jepa_encoder(config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model = SuperconductivityCrysJEPA(
        encoder,
        z_dim=int(model_cfg.get("z_dim", getattr(encoder, "z_dim", 512))),
        hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
        freeze_encoder=bool(train_cfg.get("freeze_encoder", True)),
        use_uncertainty=bool(model_cfg.get("use_uncertainty", False)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, device


def predict_cif(checkpoint_path: str | Path, cif_path: str | Path, device: str | None = None) -> dict:
    """Predict superconductivity probability and Tc for a single CIF."""
    model, _config, device_obj = load_model_for_inference(checkpoint_path, device=device)
    tensors = load_cif_tensors(cif_path)
    sample = {
        "X": tensors["X"],
        "A": tensors["A"],
        "L": tensors["L"],
        "Tc": torch.tensor(0.0),
        "label_supra": torch.tensor(0.0),
        "formula": tensors["formula_from_cif"],
        "cif_path": str(cif_path),
    }
    batch = collate_crystals([sample])
    batch = {key: value.to(device_obj) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        outputs = model(batch)
    result = {
        "material": sample["formula"],
        "prob_supra": float(torch.sigmoid(outputs["logit_supra"])[0].detach().cpu()),
        "tc": float(outputs["tc"][0].detach().cpu()),
    }
    if "sigma" in outputs:
        result["uncertainty"] = float(outputs["sigma"][0].detach().cpu())
    return result


def format_prediction(result: dict) -> str:
    """Format one inference result for the command line."""
    lines = [
        f"Material: {result['material']}",
        f"P(superconductor): {result['prob_supra']:.4f}",
        f"Predicted Tc: {result['tc']:.2f} K",
    ]
    if "uncertainty" in result:
        lines.append(f"Uncertainty: {result['uncertainty']:.2f} K")
    return "\n".join(lines)
