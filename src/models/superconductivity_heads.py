
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
    """Crys-JEPA/DFT supervised heads for superconductivity ablations."""

    VALID_INPUT_MODES = {"crys_jepa", "dft", "crys_jepa_dft"}

    def __init__(
        self,
        crys_jepa_encoder: nn.Module,
        z_dim: int,
        hidden_dim: int = 256,
        freeze_encoder: bool = True,
        use_uncertainty: bool = False,
        dropout: float = 0.1,
        input_mode: str = "crys_jepa",
        dft_feature_dim: int = 0,
        dft_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(f"Unsupported input_mode={input_mode!r}; expected one of {sorted(self.VALID_INPUT_MODES)}")
        if input_mode in {"dft", "crys_jepa_dft"} and dft_feature_dim <= 0:
            raise ValueError("dft_feature_dim must be positive when using DFT input modes")

        self.encoder = crys_jepa_encoder
        self.input_mode = input_mode
        self.use_uncertainty = use_uncertainty
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.structure_norm = nn.LayerNorm(z_dim)
        self.dft_encoder = None
        if input_mode in {"dft", "crys_jepa_dft"}:
            self.dft_encoder = nn.Sequential(
                nn.LayerNorm(dft_feature_dim),
                nn.Linear(dft_feature_dim, dft_embedding_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dft_embedding_dim, dft_embedding_dim),
                nn.SiLU(),
            )

        if input_mode == "crys_jepa":
            head_dim = z_dim
            self.fusion = None
        elif input_mode == "dft":
            head_dim = dft_embedding_dim
            self.fusion = None
        else:
            head_dim = hidden_dim
            self.fusion = nn.Sequential(
                nn.LayerNorm(z_dim + dft_embedding_dim),
                nn.Linear(z_dim + dft_embedding_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )

        self.classification_head = MLPHead(head_dim, hidden_dim, 1, dropout=dropout)
        self.regression_head = MLPHead(head_dim, hidden_dim, 1, dropout=dropout)
        self.uncertainty_head = MLPHead(head_dim, hidden_dim, 1, dropout=dropout) if use_uncertainty else None
        self.softplus = nn.Softplus()

    def _representation(self, batch: dict) -> torch.Tensor:
        if self.input_mode == "crys_jepa":
            z = self.encoder(batch["X"], batch["A"], batch["L"], batch["atom_mask"])
            return self.structure_norm(z)

        if "dft_features" not in batch:
            raise KeyError("Batch is missing dft_features required by DFT input mode")
        dft_z = self.dft_encoder(batch["dft_features"])
        if self.input_mode == "dft":
            return dft_z

        z = self.encoder(batch["X"], batch["A"], batch["L"], batch["atom_mask"])
        return self.fusion(torch.cat([self.structure_norm(z), dft_z], dim=-1))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        h = self._representation(batch)
        outputs = {
            "tc": self.regression_head(h).squeeze(-1),
            "logit_supra": self.classification_head(h).squeeze(-1),
        }
        if self.uncertainty_head is not None:
            outputs["sigma"] = self.softplus(self.uncertainty_head(h)).squeeze(-1) + 1e-6
        return outputs
