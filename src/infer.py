
from pathlib import Path

import torch

from src.datasets.threedsc_dataset import collate_crystals
from src.training.evaluate import build_model_from_config
from src.utils.cif_utils import load_cif_tensors


def load_model_for_inference(checkpoint_path: str | Path, device: str | torch.device | None = None):
    """Load a supervised checkpoint and rebuild its model."""
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, device


def predict_cif(checkpoint_path: str | Path, cif_path: str | Path, device: str | None = None) -> dict:
    """Predict superconductivity from a single CIF for structure-only checkpoints."""
    model, config, device_obj = load_model_for_inference(checkpoint_path, device=device)
    input_mode = config.get("model", {}).get("input_mode", "crys_jepa")
    if input_mode in {"dft", "crys_jepa_dft"}:
        raise ValueError(
            "This checkpoint requires DFT CSV features. Use the evaluation pipeline on a CSV row, "
            "or train/use a crys_jepa checkpoint for CIF-only inference."
        )

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
