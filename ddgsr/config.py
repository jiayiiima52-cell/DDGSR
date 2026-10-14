from dataclasses import dataclass
from experiment_config import DEFAULT_SEED


DATASET_CONFIGS = {
    "ml-100k": dict(
        users=943, items=1523, layers=3, neg_layers=3, cl_weight=0.01,
        temperature=0.5, graph_ratio=0.25, graph_weight=0.2,
        update_every=10,
    ),
    "gowalla": dict(
        users=8710, items=5855, layers=3, neg_layers=3, cl_weight=0.01,
        temperature=0.25, graph_ratio=1.0, graph_weight=0.4,
        update_every=2,
    ),
    "citeulike": dict(
        users=5551, items=16980, layers=3, neg_layers=3, cl_weight=0.002,
        temperature=0.25, graph_ratio=0.15, graph_weight=0.1,
        update_every=15,
    ),
    "programmableweb": dict(
        users=2906, items=1255, layers=3, neg_layers=3, cl_weight=0.01,
        temperature=0.5, graph_ratio=0.25, graph_weight=0.2,
        update_every=10,
    ),
    "tssgcf": dict(
        users=2289, items=956, layers=3, neg_layers=3, cl_weight=0.01,
        temperature=0.5, graph_ratio=0.25, graph_weight=0.2,
        update_every=10,
    ),
}


@dataclass
class Config:
    dataset: str = "ml-100k"
    users: int = 943
    items: int = 1523
    dim: int = 64
    layers: int = 3
    neg_layers: int = 3
    batch_size: int = 512
    epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    cl_weight: float = 0.01
    temperature: float = 0.5
    graph_ratio: float = 0.25
    graph_weight: float = 0.2
    update_every: int = 10
    patience: int = 5
    seed: int = DEFAULT_SEED

    @classmethod
    def for_dataset(cls, name: str, **overrides):
        if name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASET_CONFIGS)}")
        values = {"dataset": name, **DATASET_CONFIGS[name], **overrides}
        return cls(**values)
