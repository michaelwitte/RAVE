# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

"""
This module handles the configuration for the python project.
"""

from collections.abc import MutableMapping
from typing import Any, Dict, Generator, Tuple

import torch


def set_random_seeds(random_seed: int) -> None:
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        random_seed (int): The seed value to be used for random number generation.

    Notes:
        - Sets the seed for the following libraries if they are available:
          - `random`: Python's built-in random module.
          - `numpy`: NumPy for handling arrays and matrices.
          - `torch`: PyTorch for deep learning operations.
          - `scipy`: SciPy for scientific computing.
        - If a library is not imported in the global scope, the seed setting for that library will be skipped.

    Example:
        set_random_seeds(42)
    """
    if "random" in globals():
        random.seed(random_seed)  # type: ignore
    else:
        print("The 'random' package is not imported, skipping random seed.")

    if "np" in globals():
        np.random.seed(random_seed)  # type: ignore
    else:
        print("The 'numpy' package is not imported, skipping numpy seed.")

    if "torch" in globals():
        torch.manual_seed(random_seed)  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(random_seed)
    else:
        print("The 'torch' package is not imported, skipping torch seed.")
    if "scipy" in globals():
        scipy.random.seed(random_seed)  # type: ignore
    else:
        print("The 'scipy' package is not imported, skipping scipy seed.")
