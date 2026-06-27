# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preset reference datasets for the VSUN benchmark.

These helpers load the packaged MP2020 and related reference collections used
when evaluating generated crystals.
"""

from functools import cached_property

from eval.vsun.reference.reference_dataset import ReferenceDataset
from eval.vsun.reference.reference_dataset_serializer import LMDBGZSerializer


class ReferenceMP2020Correction(ReferenceDataset):
    """Reference dataset using the MP2020 Energy Correction scheme.
    This dataset contains entries from the Materials Project [https://next-gen.materialsproject.org/]
    and Alexandria [https://next-gen.materialsproject.org/].
    All 845,997 structures are relaxed using the GGA-PBE functional and have energy corrections applied using the MP2020 scheme.
    """

    def __init__(self):
        """Load the packaged MP2020-corrected reference dataset."""
        super().__init__("MP2020correction", ReferenceMP2020Correction.from_preset())

    @classmethod
    def from_preset(cls) -> "ReferenceMP2020Correction":
        """Deserialize the packaged MP2020-corrected dataset."""
        return LMDBGZSerializer().deserialize(
            f"eval/vsun/ref_dataset/MP2020correction/reference_MP2020correction.gz"
        )

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.

class MP2023(ReferenceDataset):
    """Reference dataset using the MP2020 Energy Correction scheme.
    This dataset contains entries from the Materials Project [https://next-gen.materialsproject.org/]
    and Alexandria [https://next-gen.materialsproject.org/].
    All 845,997 structures are relaxed using the GGA-PBE functional and have energy corrections applied using the MP2020 scheme.
    """

    def __init__(self):
        """Load the packaged MP2023 reference dataset."""
        super().__init__("mp_02072023", MP2023.from_preset())

    @classmethod
    def from_preset(cls) -> "MP2023":
        """Deserialize the packaged MP2023 dataset."""
        return LMDBGZSerializer().deserialize(f"eval/vsun/ref_dataset/mp/mp.gz")

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.
    

class alex_mp_20(ReferenceDataset):
    """Reference dataset using the MP2020 Energy Correction scheme.
    This dataset contains entries from the Materials Project [https://next-gen.materialsproject.org/]
    and Alexandria [https://next-gen.materialsproject.org/].
    All 845,997 structures are relaxed using the GGA-PBE functional and have energy corrections applied using the MP2020 scheme.
    """

    def __init__(self):
        """Load the packaged Alex-MP-20 reference dataset."""
        super().__init__("alex_mp_20", alex_mp_20.from_preset())

    @classmethod
    def from_preset(cls) -> "alex_mp_20":
        """Deserialize the packaged Alex-MP-20 dataset."""
        return LMDBGZSerializer().deserialize(f"eval/vsun/ref_dataset/alex_mp_20/alex_mp_20.gz")

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.


class mp_20(ReferenceDataset):
    """Reference dataset using the MP2020 Energy Correction scheme.
    This dataset contains entries from the Materials Project [https://next-gen.materialsproject.org/]
    and Alexandria [https://next-gen.materialsproject.org/].
    All 845,997 structures are relaxed using the GGA-PBE functional and have energy corrections applied using the MP2020 scheme.
    """

    def __init__(self):
        """Load the packaged MP-20 reference dataset."""
        super().__init__("mp_20", mp_20.from_preset())

    @classmethod
    def from_preset(cls) -> "mp_20":
        """Deserialize the packaged MP-20 dataset."""
        return LMDBGZSerializer().deserialize(f"eval/vsun/ref_dataset/mp_20/mp_20.gz")

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.
