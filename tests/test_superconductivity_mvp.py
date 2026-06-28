import csv
from pathlib import Path

import pytest
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from src.datasets.threedsc_dataset import ThreeDSCDataset, collate_crystals, split_dataset
from src.models.crys_jepa_wrapper import (
    PlaceholderCrysJEPAEncoder,
    build_crys_jepa_encoder,
    configure_jepa_partial_finetuning,
)
from src.models.superconductivity_heads import SuperconductivityCrysJEPA
from src.utils.cif_utils import load_cif_tensors
from src.utils.dft_features import DFTFeatureScaler
from src.training.train import build_optimizer
from src.utils.metrics import classification_metrics, regression_metrics


def _write_cif(path: Path, formula: str = "NaCl") -> None:
    species = ["Na", "Cl"] if formula == "NaCl" else ["Mg"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]] if formula == "NaCl" else [[0, 0, 0]]
    structure = Structure(Lattice.cubic(5.64), species, coords)
    path.write_text(str(CifWriter(structure)), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_reads_csv_and_extracts_crystal_tensors(tmp_path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "nacl.cif")
    csv_path = tmp_path / "3DSC_MP.csv"
    _write_csv(csv_path, [{"formula": "NaCl", "Tc": "12.5", "cif_path": "nacl.cif"}])

    dataset = ThreeDSCDataset(csv_path=csv_path, cif_dir=cif_dir)

    sample = dataset[0]
    assert sample["formula"] == "NaCl"
    assert sample["X"].shape == (2, 3)
    assert sample["A"].tolist() == [11, 17]
    assert sample["L"].shape == (3, 3)
    assert sample["Tc"].item() == 12.5
    assert sample["label_supra"].item() == 1.0


def test_collate_crystals_pads_variable_atom_counts():
    batch = [
        {
            "X": torch.ones(2, 3),
            "A": torch.tensor([8, 8]),
            "L": torch.eye(3),
            "Tc": torch.tensor(0.0),
            "label_supra": torch.tensor(0.0),
            "formula": "O2",
            "cif_path": "o2.cif",
        },
        {
            "X": torch.zeros(1, 3),
            "A": torch.tensor([14]),
            "L": torch.eye(3) * 2,
            "Tc": torch.tensor(90.0),
            "label_supra": torch.tensor(1.0),
            "formula": "Si",
            "cif_path": "si.cif",
        },
    ]

    collated = collate_crystals(batch)

    assert collated["X"].shape == (2, 2, 3)
    assert collated["A"].shape == (2, 2)
    assert collated["atom_mask"].tolist() == [[True, True], [True, False]]
    assert collated["Tc"].tolist() == [0.0, 90.0]
    assert collated["formula"] == ["O2", "Si"]


def test_model_returns_superconductivity_outputs_with_placeholder_encoder():
    encoder = PlaceholderCrysJEPAEncoder(z_dim=16)
    model = SuperconductivityCrysJEPA(encoder, z_dim=16, use_uncertainty=True)
    batch = {
        "X": torch.rand(3, 4, 3),
        "A": torch.tensor([[8, 8, 0, 0], [14, 0, 0, 0], [29, 8, 8, 0]]),
        "L": torch.eye(3).repeat(3, 1, 1),
        "atom_mask": torch.tensor(
            [[True, True, False, False], [True, False, False, False], [True, True, True, False]]
        ),
    }

    outputs = model(batch)

    assert outputs["tc"].shape == (3,)
    assert outputs["logit_supra"].shape == (3,)
    assert outputs["sigma"].shape == (3,)
    assert torch.all(outputs["sigma"] > 0)


def test_metrics_include_superconductor_and_high_tc_slices():
    y_true = torch.tensor([0.0, 1.0, 1.0, 0.0])
    logits = torch.tensor([-4.0, 3.0, -1.0, 2.0])
    tc_true = torch.tensor([0.0, 20.0, 100.0, 0.0])
    tc_pred = torch.tensor([1.0, 18.0, 90.0, 5.0])

    cls = classification_metrics(logits, y_true)
    reg = regression_metrics(tc_pred, tc_true, high_tc_threshold=77.0)

    assert cls["accuracy"] == 0.5
    assert cls["confusion_matrix"] == [[1, 1], [1, 1]]
    assert reg["mae"] == 4.5
    assert reg["mae_superconductors"] == 6.0
    assert reg["mae_high_tc"] == 10.0


def test_split_dataset_is_reproducible(tmp_path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    rows = []
    for idx in range(10):
        name = f"mat_{idx}.cif"
        _write_cif(cif_dir / name)
        rows.append({"formula": f"NaCl{idx}", "Tc": str(float(idx)), "cif_path": name})
    csv_path = tmp_path / "3DSC_MP.csv"
    _write_csv(csv_path, rows)
    dataset = ThreeDSCDataset(csv_path=csv_path, cif_dir=cif_dir)

    first = split_dataset(dataset, train=0.6, val=0.2, test=0.2, seed=123)
    second = split_dataset(dataset, train=0.6, val=0.2, test=0.2, seed=123)

    assert [subset.indices for subset in first] == [subset.indices for subset in second]
    assert [len(subset) for subset in first] == [6, 2, 2]



def test_dataset_ignores_comment_lines_and_resolves_existing_csv_cif_paths(tmp_path):
    cif_dir = tmp_path / "unused_cifs"
    cif_dir.mkdir()
    real_cif = tmp_path / "existing.cif"
    _write_cif(real_cif)
    csv_path = tmp_path / "3DSC_MP.csv"
    csv_path.write_text(
        "# generated metadata line\n"
        "formula_sc,tc,cif\n"
        f"NaCl,4.2,{real_cif}\n",
        encoding="utf-8",
    )

    dataset = ThreeDSCDataset(
        csv_path=csv_path,
        cif_dir=cif_dir,
        formula_column="formula_sc",
        tc_column="tc",
        cif_column="cif",
        csv_comment="#",
    )

    sample = dataset[0]
    assert sample["formula"] == "NaCl"
    assert sample["Tc"].item() == pytest.approx(4.2)
    assert sample["cif_path"] == str(real_cif)



def test_dataset_falls_back_to_flat_cif_dir_when_csv_contains_historical_path(tmp_path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "material.cif")
    csv_path = tmp_path / "3DSC_MP.csv"
    csv_path.write_text(
        "formula_sc,tc,cif\n"
        "NaCl,5.0,data/final/MP/cifs/material.cif\n",
        encoding="utf-8",
    )

    dataset = ThreeDSCDataset(
        csv_path=csv_path,
        cif_dir=cif_dir,
        formula_column="formula_sc",
        tc_column="tc",
        cif_column="cif",
    )

    sample = dataset[0]
    assert sample["cif_path"] == str(cif_dir / "material.cif")



def test_load_cif_tensors_uses_majority_species_for_disordered_sites(tmp_path):
    cif_path = tmp_path / "disordered.cif"
    structure = Structure(
        Lattice.cubic(4.0),
        [{"Ag": 0.25, "Pd": 0.75}, "Sr"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    cif_path.write_text(str(CifWriter(structure)), encoding="utf-8")

    tensors = load_cif_tensors(cif_path)

    assert sorted(tensors["A"].tolist()) == [38, 46]
    assert 47 not in tensors["A"].tolist()
    assert tensors["X"].shape == (2, 3)



def test_dft_feature_scaler_fits_train_values_and_imputes_nan():
    train_values = torch.tensor(
        [
            [1.0, float("nan")],
            [3.0, 10.0],
            [5.0, 14.0],
        ]
    )

    scaler = DFTFeatureScaler.fit(train_values)
    transformed = scaler.transform(torch.tensor([[float("nan"), 10.0]]))

    assert scaler.median.tolist() == pytest.approx([3.0, 12.0])
    assert transformed.shape == (1, 2)
    assert torch.isfinite(transformed).all()
    assert transformed[0, 0].item() == pytest.approx(0.0)


def test_dataset_returns_scaled_dft_features_from_configured_columns(tmp_path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "nacl.cif")
    csv_path = tmp_path / "3DSC_MP.csv"
    _write_csv(
        csv_path,
        [
            {"formula": "NaCl", "Tc": "12.5", "cif_path": "nacl.cif", "band_gap_2": "1.5", "density_2": ""},
        ],
    )
    scaler = DFTFeatureScaler.fit(torch.tensor([[1.5, float("nan")], [2.5, 4.0]]))
    dataset = ThreeDSCDataset(
        csv_path=csv_path,
        cif_dir=cif_dir,
        dft_feature_columns=["band_gap_2", "density_2"],
        dft_scaler=scaler,
    )

    sample = dataset[0]

    assert sample["dft_features"].shape == (2,)
    assert torch.isfinite(sample["dft_features"]).all()


def test_model_supports_crys_jepa_dft_and_dft_only_modes():
    batch = {
        "X": torch.rand(2, 3, 3),
        "A": torch.tensor([[8, 8, 0], [14, 0, 0]]),
        "L": torch.eye(3).repeat(2, 1, 1),
        "atom_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "dft_features": torch.randn(2, 4),
    }

    fusion_model = SuperconductivityCrysJEPA(
        PlaceholderCrysJEPAEncoder(z_dim=8),
        z_dim=8,
        input_mode="crys_jepa_dft",
        dft_feature_dim=4,
        dft_embedding_dim=6,
        use_uncertainty=True,
    )
    dft_model = SuperconductivityCrysJEPA(
        PlaceholderCrysJEPAEncoder(z_dim=8),
        z_dim=8,
        input_mode="dft",
        dft_feature_dim=4,
        dft_embedding_dim=6,
    )

    fusion_outputs = fusion_model(batch)
    dft_outputs = dft_model(batch)

    assert fusion_outputs["tc"].shape == (2,)
    assert fusion_outputs["sigma"].shape == (2,)
    assert dft_outputs["logit_supra"].shape == (2,)



def test_partial_finetuning_unfreezes_only_last_jepa_blocks():
    class FakeJEPA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_backbone = torch.nn.Linear(2, 2)
            self.backbone = torch.nn.Module()
            self.backbone.block = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
            self.backbone.norm = torch.nn.LayerNorm(2)
            self.predictor = torch.nn.Linear(2, 2)
            self.cond_emb = torch.nn.Linear(2, 2)

    fake = FakeJEPA()
    trainable = configure_jepa_partial_finetuning(fake, unfreeze_last_n_layers=1)

    assert trainable > 0
    assert all(not param.requires_grad for param in fake.pre_backbone.parameters())
    assert all(not param.requires_grad for param in fake.backbone.block[0].parameters())
    assert all(not param.requires_grad for param in fake.backbone.block[1].parameters())
    assert all(param.requires_grad for param in fake.backbone.block[2].parameters())
    assert all(param.requires_grad for param in fake.backbone.norm.parameters())
    assert all(not param.requires_grad for param in fake.predictor.parameters())
    assert all(not param.requires_grad for param in fake.cond_emb.parameters())


def test_build_optimizer_uses_separate_encoder_learning_rate():
    encoder = PlaceholderCrysJEPAEncoder(z_dim=8)
    model = SuperconductivityCrysJEPA(
        encoder,
        z_dim=8,
        freeze_encoder=False,
        input_mode="crys_jepa",
    )
    config = {
        "training": {
            "learning_rate": 1e-4,
            "encoder_learning_rate": 1e-5,
            "weight_decay": 1e-4,
        }
    }

    optimizer = build_optimizer(model, config)

    assert len(optimizer.param_groups) == 2
    assert sorted(group["lr"] for group in optimizer.param_groups) == [1e-5, 1e-4]
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        [param for param in model.parameters() if param.requires_grad]
    )


def test_build_crys_jepa_encoder_reports_missing_checkpoint(tmp_path):
    missing_checkpoint = tmp_path / "missing_pretrained.pt"
    config = {
        "model": {
            "crys_jepa_checkpoint": str(missing_checkpoint),
            "crys_jepa_config": "configs/jepa/mp.yml",
        }
    }

    with pytest.raises(FileNotFoundError, match="Crys-JEPA checkpoint not found"):
        build_crys_jepa_encoder(config)
