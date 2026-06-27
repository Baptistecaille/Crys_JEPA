from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from src.utils.cif_utils import load_cif_tensors


class ThreeDSCDataset(Dataset):
    """PyTorch dataset for 3DSC/3DSCMP rows linked to CIF files."""

    def __init__(
        self,
        csv_path: str | Path,
        cif_dir: str | Path,
        formula_column: str = "formula",
        tc_column: str = "Tc",
        cif_column: str = "cif_path",
        csv_comment: str | None = None,
        primitive: bool = False,
        reduced: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.cif_dir = Path(cif_dir)
        self.formula_column = formula_column
        self.tc_column = tc_column
        self.cif_column = cif_column
        self.csv_comment = csv_comment
        self.primitive = primitive
        self.reduced = reduced
        self.rows = pd.read_csv(self.csv_path, comment=self.csv_comment)
        self._validate_columns()

    def _validate_columns(self) -> None:
        missing = [
            column
            for column in (self.tc_column, self.cif_column)
            if column not in self.rows.columns
        ]
        if missing:
            raise ValueError(f"Missing required 3DSC CSV columns: {missing}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        tc = float(row[self.tc_column])
        cif_path = self._resolve_cif_path(row[self.cif_column])
        tensors = load_cif_tensors(cif_path, primitive=self.primitive, reduced=self.reduced)
        formula = str(row[self.formula_column]) if self.formula_column in row else tensors["formula_from_cif"]

        return {
            "X": tensors["X"],
            "A": tensors["A"],
            "L": tensors["L"],
            "Tc": torch.tensor(tc, dtype=torch.float32),
            "label_supra": torch.tensor(1.0 if tc > 0.0 else 0.0, dtype=torch.float32),
            "formula": formula,
            "cif_path": str(cif_path),
        }

    def _resolve_cif_path(self, value: object) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        if path.exists():
            return path
        nested_path = self.cif_dir / path
        if nested_path.exists():
            return nested_path
        flat_path = self.cif_dir / path.name
        if flat_path.exists():
            return flat_path
        return nested_path


def collate_crystals(batch: Sequence[dict]) -> dict:
    """Pad atom tensors and build a boolean atom mask for variable-size crystals."""
    if not batch:
        raise ValueError("Cannot collate an empty crystal batch")

    batch_size = len(batch)
    max_atoms = max(item["X"].shape[0] for item in batch)
    x = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    a = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    atom_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)

    for idx, item in enumerate(batch):
        n_atoms = item["X"].shape[0]
        x[idx, :n_atoms] = item["X"].float()
        a[idx, :n_atoms] = item["A"].long()
        atom_mask[idx, :n_atoms] = True

    return {
        "X": x,
        "A": a,
        "L": torch.stack([item["L"].float() for item in batch], dim=0),
        "atom_mask": atom_mask,
        "Tc": torch.stack([item["Tc"].float() for item in batch]),
        "label_supra": torch.stack([item["label_supra"].float() for item in batch]),
        "formula": [item["formula"] for item in batch],
        "cif_path": [item["cif_path"] for item in batch],
    }


def split_dataset(dataset: Dataset, train: float, val: float, test: float, seed: int) -> tuple[Subset, Subset, Subset]:
    """Create reproducible train/validation/test subsets."""
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    n_items = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_items, generator=generator).tolist()
    n_train = int(n_items * train)
    n_val = int(n_items * val)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)
