import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> int:
    """Set RNG seeds for reproducibility.

    Returns the seed actually used so it can be logged.
    """
    if seed is None:
        seed = int(os.environ.get("PYTHONHASHSEED", 0))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

