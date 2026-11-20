import os
import random

import numpy as np
import torch

SEED = 6304  # one seed for every RNG in the project (guide §5, global_seed)


def seed_everything(seed: int = SEED) -> None:
    """Seed python, numpy, torch CPU and every CUDA device."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
