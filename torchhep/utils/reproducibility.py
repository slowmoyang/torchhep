import os
import random
import numpy as np
import torch

def sample_seed():
    return int.from_bytes(os.urandom(4), byteorder="big")

def set_seed(seed: int):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
