from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict

from src.utils.cif_utils import lattice_to_symmetric_features


class IdentityScaler:
    """Fallback matrix scaler used when Crys-JEPA scaling statistics are unavailable."""

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class PlaceholderCrysJEPAEncoder(nn.Module):
    """Small compatible encoder used until a pretrained Crys-JEPA checkpoint is supplied."""

    def __init__(self, z_dim: int = 512, max_atomic_number: int = 100, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or z_dim
        self.z_dim = z_dim
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden, padding_idx=0)
        self.coord_mlp = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.lattice_mlp = nn.Sequential(nn.Linear(9, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, z_dim), nn.SiLU())

    def forward(self, X: torch.Tensor, A: torch.Tensor, L: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        atom_emb = self.atom_embedding(A.clamp(min=0, max=self.atom_embedding.num_embeddings - 1))
        coord_emb = self.coord_mlp(X)
        tokens = (atom_emb + coord_emb) * atom_mask.unsqueeze(-1)
        denom = atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
        pooled_atoms = tokens.sum(dim=1) / denom
        lattice = self.lattice_mlp(L.reshape(L.shape[0], 9))
        return self.out(torch.cat([pooled_atoms, lattice], dim=-1))


class CrysJEPAEncoderAdapter(nn.Module):
    """Adapter exposing the pretrained Crys-JEPA encoder as forward(X, A, L, atom_mask)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path = "configs/jepa/mp.yml",
        matrix_scaler: object | None = None,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        from components.jepa.frame.jepa import JEPA

        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = EasyDict(yaml.safe_load(handle))
        config.device = "cpu"
        self.matrix_scaler = matrix_scaler or IdentityScaler()
        self.encoder = JEPA(config, matrix_scaler=self.matrix_scaler)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
        self.encoder.load_state_dict(state, strict=False)
        self.z_dim = int(config.model.hidden_dim)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward(self, X: torch.Tensor, A: torch.Tensor, L: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with context:
            return self._encode(X, A, L, atom_mask)

    def _encode(self, X: torch.Tensor, A: torch.Tensor, L: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        bsz, max_atoms, _ = X.shape
        device = X.device
        num_atoms = atom_mask.sum(dim=1).long().clamp_min(1)
        lattice_features = lattice_to_symmetric_features(L) / num_atoms.unsqueeze(1).float().pow(1.0 / 3.0)
        lattice_features = self.matrix_scaler.transform(lattice_features)

        flat_mask = atom_mask.reshape(-1)
        flat_x = X.reshape(bsz * max_atoms, 3)[flat_mask]
        flat_a = A.reshape(bsz * max_atoms)[flat_mask].long().clamp(min=0, max=99)
        batch = torch.repeat_interleave(torch.arange(bsz, device=device), num_atoms)
        token_features = torch.cat(
            [flat_x.float(), F.one_hot(flat_a, 100).float(), lattice_features[batch].float()],
            dim=-1,
        )
        return self.encoder.encode(token_features, batch)


def build_crys_jepa_encoder(config: dict) -> nn.Module:
    """Build a pretrained Crys-JEPA adapter when possible, otherwise a placeholder."""
    model_cfg = config.get("model", {})
    z_dim = int(model_cfg.get("z_dim", 512))
    checkpoint_path = model_cfg.get("crys_jepa_checkpoint")
    if checkpoint_path:
        return CrysJEPAEncoderAdapter(
            checkpoint_path=checkpoint_path,
            config_path=model_cfg.get("crys_jepa_config", "configs/jepa/mp.yml"),
            freeze_encoder=bool(config.get("training", {}).get("freeze_encoder", True)),
        )
    return PlaceholderCrysJEPAEncoder(z_dim=z_dim)
