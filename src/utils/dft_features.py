
"""Scaling helpers for optional DFT feature inputs."""

import numpy as np
import torch


class DFTFeatureScaler:
    """Train-split scaler for numerical DFT features with median imputation."""

    def __init__(self, median: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store imputation and standardization statistics."""
        self.median = median.float()
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-8)

    @classmethod
    def fit(cls, values: torch.Tensor) -> "DFTFeatureScaler":
        """Fit imputation and standardization statistics from train values only."""
        values = values.float()
        if values.ndim != 2:
            raise ValueError(f"DFT feature values must be 2D, got shape {tuple(values.shape)}")
        median_np = np.nanmedian(values.detach().cpu().numpy(), axis=0)
        median_np = np.where(np.isnan(median_np), 0.0, median_np)
        median = torch.tensor(median_np, dtype=torch.float32, device=values.device)
        imputed = torch.where(torch.isnan(values), median.unsqueeze(0), values)
        mean = imputed.mean(dim=0)
        std = imputed.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return cls(median=median.cpu(), mean=mean.cpu(), std=std.cpu())

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Impute missing values and return standardized DFT features."""
        values = values.float()
        squeeze = values.ndim == 1
        if squeeze:
            values = values.unsqueeze(0)
        median = self.median.to(values.device).unsqueeze(0)
        mean = self.mean.to(values.device).unsqueeze(0)
        std = self.std.to(values.device).unsqueeze(0)
        imputed = torch.where(torch.isnan(values), median, values)
        transformed = (imputed - mean) / std
        return transformed.squeeze(0) if squeeze else transformed

    def state_dict(self) -> dict[str, list[float]]:
        """Return JSON/YAML-friendly scaler statistics."""
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DFTFeatureScaler":
        """Restore a scaler from checkpoint/config state."""
        return cls(
            median=torch.tensor(state["median"], dtype=torch.float32),
            mean=torch.tensor(state["mean"], dtype=torch.float32),
            std=torch.tensor(state["std"], dtype=torch.float32),
        )
