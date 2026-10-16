import numpy as np
import scipy.sparse as sp
import torch

from ddgsrl.config import Config
from ddgsrl.model import DDGSRL, scipy_to_torch


def test_forward_and_loss_are_finite():
    cfg = Config(users=3, items=4, dim=8, layers=2, neg_layers=2)
    model = DDGSRL(cfg)
    rating = sp.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 4))
    adjacency = sp.bmat([[None, rating], [rating.T, None]], format="csr", dtype=np.float32)
    graph = scipy_to_torch(adjacency, "cpu")
    loss, _, _ = model.loss(torch.tensor([0, 1]), torch.tensor([0, 1]),
                            torch.tensor([2, 3]), (graph, -graph))
    assert torch.isfinite(loss)
    assert model.scores((graph, -graph)).shape == (3, 4)

