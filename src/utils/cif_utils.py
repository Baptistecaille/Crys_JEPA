
"""CIF loading and lattice feature conversion helpers."""

from pathlib import Path

import torch
from pymatgen.core import Structure


def _site_atomic_number(site) -> int:
    """Return one atomic number for ordered or partially occupied pymatgen sites."""
    try:
        return int(site.specie.Z)
    except AttributeError:
        pass

    if not site.species:
        raise ValueError(f"Site has no species: {site}")
    species, _occupancy = max(site.species.items(), key=lambda item: float(item[1]))
    if not hasattr(species, "Z"):
        raise ValueError(f"Cannot map non-element species to an atomic number: {species}")
    return int(species.Z)


def _structure_atomic_numbers(structure: Structure) -> list[int]:
    """Convert ordered or disordered structure sites to atomic numbers."""
    return [_site_atomic_number(site) for site in structure]


def load_cif_tensors(cif_path: str | Path, primitive: bool = False, reduced: bool = False) -> dict[str, torch.Tensor]:
    """Load one CIF file and return fractional coordinates, atomic numbers, and lattice."""
    structure = Structure.from_file(str(cif_path))
    if primitive:
        structure = structure.get_primitive_structure()
    if reduced:
        structure = structure.get_reduced_structure()

    return {
        "X": torch.tensor(structure.frac_coords.copy(), dtype=torch.float32),
        "A": torch.tensor(_structure_atomic_numbers(structure), dtype=torch.long),
        "L": torch.tensor(structure.lattice.matrix.copy(), dtype=torch.float32),
        "formula_from_cif": structure.composition.reduced_formula,
    }


def lattice_to_symmetric_features(lattice: torch.Tensor) -> torch.Tensor:
    """Match the compact six-value lattice representation used by Crys-JEPA."""
    w, s, vh = torch.linalg.svd(lattice)
    s_square = torch.diag_embed(s)
    v = vh.transpose(-2, -1)
    p = v @ s_square @ vh
    u = w @ vh
    sym = u @ p @ u.transpose(-2, -1)
    sym = torch.where(torch.abs(sym) < 1e-5, torch.zeros_like(sym), sym)
    tri = torch.triu_indices(3, 3, device=lattice.device)
    return sym[..., tri[0], tri[1]]
