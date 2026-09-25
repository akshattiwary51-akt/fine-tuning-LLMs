"""Reproducibility helpers: seed Python, NumPy and PyTorch RNGs consistently."""
from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seeds every RNG this pipeline touches (dataset shuffling included, since
    `datasets`/`numpy` share the same underlying RNG family as numpy.random)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.debug("NumPy not installed; skipping NumPy seeding.")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("PyTorch not installed; skipping PyTorch seeding.")

    logger.info("Random seed set to %d (python, numpy, torch).", seed)
