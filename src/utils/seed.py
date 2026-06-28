"""Reproducibility utilities shared by training scripts."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
