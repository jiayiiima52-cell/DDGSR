"""Shared experimental protocol for DDGSRL and baseline runners."""
SEEDS = (42, 100, 2024, 3407, 8080)
DEFAULT_SEED = SEEDS[0]
SPLIT_SEED = 100
GRAPH_SEED = 100
EVAL_KS = (5, 10, 15, 20)


def seed_everything(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
