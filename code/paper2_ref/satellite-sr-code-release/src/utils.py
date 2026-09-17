import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Fix all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
