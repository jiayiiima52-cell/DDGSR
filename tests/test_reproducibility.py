import torch
from ddgsrl.config import Config
from ddgsrl.model import DDGSRL
from experiment_config import SEEDS
from run_experiment import command


def build(seed):
    return DDGSRL(Config(users=3, items=4, dim=8, seed=seed))


def test_initialization_follows_run_seed():
    first = build(SEEDS[0])
    repeated = build(SEEDS[0])
    other = build(SEEDS[1])
    for name in ("pos_user", "pos_item", "neg_user", "neg_item"):
        left = getattr(first, name).weight
        assert torch.equal(left, getattr(repeated, name).weight)
        assert not torch.equal(left, getattr(other, name).weight)


def test_runner_passes_each_protocol_seed(tmp_path):
    for seed in SEEDS:
        args = command(seed, tmp_path / "data", tmp_path / "outputs", "cpu")
        assert args[args.index("--seed") + 1] == str(seed)
        assert args[args.index("--variant") + 1] == "ddgsrl"
