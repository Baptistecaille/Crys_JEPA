# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build summary entries from relaxed structures and energies.

This module turns relaxed outputs into computed entries compatible with the
Materials Project correction scheme used by the VSUN evaluation pipeline.
"""

from dataclasses import dataclass, field
from functools import cached_property
import re

from pymatgen.core import Structure
from pymatgen.entries.compatibility import Compatibility, MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.structure_analyzer import oxide_type
from pymatgen.core import Structure
from pymatgen.entries.compatibility import Compatibility
from pymatgen.entries.computed_entries import (
    ComputedEntry,
    ComputedStructureEntry,
    EnergyAdjustment,
)
from pymatgen.io.vasp.outputs import VaspParseError
from pymatgen.io.vasp.sets import MPRelaxSet


class IdentityCorrectionScheme(Compatibility):
    """Perform no energy correction."""

    def get_adjustments(
        self, entry: ComputedEntry | ComputedStructureEntry
    ) -> list[EnergyAdjustment]:
        """Return no corrections for the supplied entry."""
        return []


class VasprunLike:
    """
    Mocks a VASP run using only the structure as well as INCAR and POTCAR information from MPRelaxSet.
    Code adapted from https://github.com/materialsproject/pymatgen/blob/6c23d744efbd892ec48346297d61b4f3f86b1478/pymatgen/io/vasp/outputs.py#L153

    Note that this object does not have the full functionality of a Vasprun. It is only used to obtain energy corrections if the full Vasprun information is not available.
    """

    def __init__(
        self, structure: Structure, energy: float, user_potcar_functional: str = "PBE"
    ) -> None:
        """Store the structure and energy needed to mimic a VASP run."""
        self.structure = structure
        self.energy = energy
        self.user_potcar_functional = user_potcar_functional

    @cached_property
    def mp_set(self) -> MPRelaxSet:
        """Build the MPRelaxSet used to infer VASP-style metadata."""
        return MPRelaxSet(
            self.structure,
            # These settings prevent the MPRelaxSet from trying to
            # automatically determine kpoints, which sometimes results
            # in SpacegroupAnalyzer errors
            user_incar_settings={"KSPACING": 0.5},
            user_kpoints_settings=None,
        )

    @property
    def potcar_symbols(self) -> list[str]:
        """Return the POTCAR symbols implied by the stored structure."""
        try:
            return [
                f"{self.user_potcar_functional.upper()} {sym}" for sym in self.mp_set.potcar_symbols
            ]
        except:
            return []

    @property
    def aspherical(self) -> bool:
        """Report whether the run uses aspherical corrections."""
        return self.mp_set.incar.get("LASPH", False)

    @property
    def hubbards(self) -> dict:
        """Return the Hubbard-U values inferred from the relaxed structure."""
        try:
            symbols = [s.split()[1] for s in self.potcar_symbols]
        except:
            return {}
        symbols = [re.split(r"_", s)[0] for s in symbols]
        if not self.mp_set.incar.get("LDAU", False):
            return {}
        us = self.mp_set.incar.get("LDAUU", [])
        js = self.mp_set.incar.get("LDAUJ", [])
        if len(js) != len(us):
            js = [0] * len(us)
        if len(us) == len(symbols):
            return {symbols[i]: us[i] - js[i] for i in range(len(symbols))}
        if sum(us) == 0 and sum(js) == 0:
            return {}
        raise VaspParseError("Length of U value parameters and atomic symbols are mismatched")

    @property
    def run_type(self) -> str:
        """Return the VASP run type implied by the inferred Hubbard settings."""

        rt = "GGA"
        if self.is_hubbard:
            rt += "+U"

        return rt

    @property
    def is_hubbard(self) -> bool:
        """Return True when the structure corresponds to a DFT+U run."""
        if len(self.hubbards) == 0:
            return False
        return sum(self.hubbards.values()) > 1e-8

    def get_computed_entry(
        self,
        inc_structure: bool = True,
        energy_correction_scheme: Compatibility = IdentityCorrectionScheme(),
    ) -> ComputedEntry:
        """Build a ComputedEntry or ComputedStructureEntry with corrections applied."""
        entry_dict = {
            "correction": 0.0,
            "composition": self.structure.composition,
            "energy": self.energy,
            "parameters": {
                "is_hubbard": self.is_hubbard,
                "hubbards": self.hubbards,
                "run_type": self.run_type,
                "potcar_symbols": self.potcar_symbols,
            },
            "data": {"oxide_type": oxide_type(self.structure), "aspherical": self.aspherical},
            "structure": self.structure,
        }

        if not inc_structure:
            entry = ComputedEntry.from_dict(entry_dict)
        else:
            entry = ComputedStructureEntry.from_dict(entry_dict)

        energy_correction_scheme.process_entry(entry)

        return entry
    
@dataclass
class MetricsStructureSummary:
    entry: ComputedStructureEntry
    properties: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_structure_and_energy(
        structure: Structure,
        energy: float,
        properties: dict[str, float] | None = None,
        energy_correction_scheme: Compatibility = MaterialsProject2020Compatibility(),
    ) -> "MetricsStructureSummary":
        """Create a summary entry from a relaxed structure and its energy."""
        vasprun_like = VasprunLike(structure=structure, energy=energy)
        entry = vasprun_like.get_computed_entry(
            inc_structure=True, energy_correction_scheme=energy_correction_scheme
        )

        return MetricsStructureSummary(
            entry=entry,
            properties=properties or {},
        )

    @property
    def structure(self) -> Structure:
        """Return the relaxed structure stored in the summary."""
        return self.entry.structure

    @property
    def chemical_system(self) -> str:
        """Return the chemical system string for the summary entry."""
        return self.entry.composition.chemical_system


def get_metrics_structure_summaries(
    structures: list[Structure],
    energies: list[float],
    energy_correction_scheme: Compatibility = MaterialsProject2020Compatibility(),
) -> list[MetricsStructureSummary]:
    """Convert parallel lists of structures and energies into summaries."""
    return [
        MetricsStructureSummary.from_structure_and_energy(
            structure=structures[i],
            energy=energies[i],
            energy_correction_scheme=energy_correction_scheme,
        )
        for i in range(len(structures))
    ]
