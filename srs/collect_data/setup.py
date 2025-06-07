# utils.py
import os
import torch
import random
import numpy as np

def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # Set CuBLAS deterministic behavior to enforce deterministic behavior for CuBLAS operations when using optuna
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
