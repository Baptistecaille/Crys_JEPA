"""Tests for robust checkpoint creation during supervised training."""

from pathlib import Path

import torch

from src.training import train as train_module


class TinyModel(torch.nn.Module):
    """Minimal trainable model used to isolate checkpoint logic."""

    def __init__(self) -> None:
        """Create one parameter consumed by optimizer/checkpoint code."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch):
        """Return fixed-shape outputs expected by evaluation stubs."""
        return {"tc": self.weight, "logit_supra": self.weight}


def test_train_from_config_creates_checkpoint_parent_and_saves_best(tmp_path, monkeypatch):
    """Verify missing checkpoint directories are created before saving best.pt."""
    checkpoint_dir = tmp_path / "missing" / "checkpoints"
    config = {
        "training": {
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "seed": 1,
            "device": "cpu",
        },
        "checkpoints": {
            "save_dir": str(checkpoint_dir),
            "best_name": "nested/best.pt",
        },
        "evaluation": {"high_tc_threshold": 77.0},
    }
    metrics = {
        "regression": {"mae": 0.25},
        "classification": {"f1": 1.0},
    }

    monkeypatch.setattr(train_module, "build_loaders", lambda _config: (object(), object(), object()))
    monkeypatch.setattr(train_module, "build_model_from_config", lambda _config: TinyModel())
    monkeypatch.setattr(train_module, "train_one_epoch", lambda *args, **kwargs: {"loss": 0.1, "loss_cls": 0.1, "loss_tc": 0.1})
    monkeypatch.setattr(train_module, "evaluate_model", lambda *args, **kwargs: metrics)

    result = train_module.train_from_config(config)

    best_path = checkpoint_dir / "nested" / "best.pt"
    assert Path(result["best_checkpoint"]) == best_path
    assert best_path.exists()
    assert not (best_path.parent / ".best.pt.tmp").exists()
