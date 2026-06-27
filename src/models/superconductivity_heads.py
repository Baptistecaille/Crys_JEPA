import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Small MLP head for supervised prediction from crystal embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SuperconductivityCrysJEPA(nn.Module):
    """Frozen Crys-JEPA encoder plus supervised heads for superconductivity."""

    def __init__(
        self,
        crys_jepa_encoder: nn.Module,
        z_dim: int,
        hidden_dim: int = 256,
        freeze_encoder: bool = True,
        use_uncertainty: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = crys_jepa_encoder
        self.use_uncertainty = use_uncertainty
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classification_head = MLPHead(z_dim, hidden_dim, 1, dropout=dropout)
        self.regression_head = MLPHead(z_dim, hidden_dim, 1, dropout=dropout)
        self.uncertainty_head = MLPHead(z_dim, hidden_dim, 1, dropout=dropout) if use_uncertainty else None
        self.softplus = nn.Softplus()

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        z = self.encoder(batch["X"], batch["A"], batch["L"], batch["atom_mask"])
        outputs = {
            "tc": self.regression_head(z).squeeze(-1),
            "logit_supra": self.classification_head(z).squeeze(-1),
        }
        if self.uncertainty_head is not None:
            outputs["sigma"] = self.softplus(self.uncertainty_head(z)).squeeze(-1) + 1e-6
        return outputs
