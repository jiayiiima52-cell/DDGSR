from __future__ import annotations

import pickle
from experiment_config import SPLIT_SEED, GRAPH_SEED
from pathlib import Path

import numpy as np
import scipy.sparse as sp


PREFIXES = {
    "ml-100k": "ml-100k",
    "gowalla": "gowalla",
    "citeulike": "citeulike",
    "programmableweb": "programmableweb",
    "tssgcf": "tssgcf",
}


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def locate_dataset(root: str | Path, dataset: str) -> Path:
    root = Path(root)
    prefix = PREFIXES[dataset]
    marker = f"{prefix}_gcn_train_data_index_from_0.pkl"
    matches = list(root.rglob(marker))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {marker} below {root}, found {len(matches)}. "
            "See README.md for the required layout."
        )
    return matches[0].parent


def load_dataset(root: str | Path, dataset: str, full_train: bool = False,
                 load_test: bool = True):
    if dataset == "tssgcf":
        root_path = Path(root)
        direct = root_path / "tssgcf_official" / "Data" / "train.txt"
        # Prefer the dataset explicitly rooted by the caller. This keeps an
        # archived confirmatory split elsewhere below ``data`` from making the
        # default official split ambiguous.
        matches = [direct] if direct.exists() else list(
            root_path.rglob("tssgcf_official/Data/train.txt")
        )
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one TSSGCF Data/train.txt below {root}, found {len(matches)}")
        folder = matches[0].parent
        train_all = _load_interactions(folder / "train.txt")
        test = _load_interactions(folder / "test.txt") if load_test else None
        train, val = (train_all, train_all) if full_train else _leave_one_out_validation(train_all, seed=SPLIT_SEED)
        interest = _interaction_graph(train, 2289, 956)
        # Build both graph views from the effective training split.  Using
        # ``train_all`` here during validation would reveal that held-out
        # validation interactions are known positives by excluding them from
        # negative-edge sampling.
        disinterest = _sampled_negative_graph(train, 2289, 956, seed=GRAPH_SEED)
        return train, val, test, interest, disinterest, folder
    if dataset == "programmableweb":
        matches = list(Path(root).rglob("programmableweb-onlyInteraction/train.txt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one ProgrammableWeb train.txt below {root}, found {len(matches)}")
        folder = matches[0].parent
        train_all = _load_interactions(folder / "train.txt")
        test = _load_interactions(folder / "test.txt") if load_test else None
        train, val = _leave_one_out_validation(train_all, seed=SPLIT_SEED)
        users = max(train_all["users"]) + 1
        items = max(train_all["pos_item"]) + 1
        if test is not None:
            users = max(users, max(test["users"]) + 1)
            items = max(items, max(test["pos_item"]) + 1)
        interest = _interaction_graph(train, users, items)
        # Keep validation interactions isolated from every graph-construction
        # input, including the negative/disinterest view.
        disinterest = _sampled_negative_graph(train, users, items, seed=GRAPH_SEED)
        return train, val, test, interest, disinterest, folder
    folder = locate_dataset(root, dataset)
    prefix = PREFIXES[dataset]
    train = _load_pickle(folder / f"{prefix}_gcn_train_data_index_from_0.pkl")
    val = _load_pickle(folder / f"{prefix}_gcn_val_data_index_from_0.pkl")
    test = (_load_pickle(folder / f"{prefix}_gcn_test_data_index_from_0.pkl")
            if load_test else None)
    interest = sp.load_npz(folder / f"{prefix}_s_pre_adj_mat_index_from_0.npz").tocsr()
    disinterest = sp.load_npz(folder / f"{prefix}_s_l_norm_3_prob_mat.npz").tocsr()
    # The official notebooks subtract I: this turns the normalized adjacency into
    # the negative propagation operator used by Eq. (15).
    disinterest = disinterest - sp.eye(disinterest.shape[0], dtype=disinterest.dtype, format="csr")
    return train, val, test, interest.astype(np.float32), disinterest.astype(np.float32), folder


def _load_interactions(path: Path):
    users, items = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [int(value) for value in line.split()]
        users.extend([fields[0]] * (len(fields) - 1))
        items.extend(fields[1:])
    return {"users": users, "pos_item": items}


def _leave_one_out_validation(split, seed: int):
    rng = np.random.default_rng(seed)
    grouped = {}
    for user, item in zip(split["users"], split["pos_item"]):
        grouped.setdefault(user, []).append(item)
    train_u, train_i, val_u, val_i = [], [], [], []
    for user, items in grouped.items():
        if len(items) > 1:
            index = int(rng.integers(len(items)))
            val_u.append(user); val_i.append(items[index])
            items = items[:index] + items[index + 1:]
        train_u.extend([user] * len(items)); train_i.extend(items)
    return {"users": train_u, "pos_item": train_i}, {"users": val_u, "pos_item": val_i}


def _normalize_bipartite(rating):
    adjacency = sp.bmat([[None, rating], [rating.T, None]], format="csr", dtype=np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.zeros_like(degree, dtype=np.float32)
    inv[degree > 0] = degree[degree > 0] ** -0.5
    scale = sp.diags(inv)
    return (scale @ adjacency @ scale).tocsr().astype(np.float32)


def _interaction_graph(split, users, items):
    us, it = arrays(split)
    rating = sp.csr_matrix((np.ones(len(us), dtype=np.float32), (us, it)), shape=(users, items))
    return _normalize_bipartite(rating)


def _sampled_negative_graph(split, users, items, seed):
    rng = np.random.default_rng(seed)
    positive = seen_items(split, users)
    rows, cols = [], []
    for user in range(users):
        count = max(1, len(positive[user]))
        available = np.asarray(list(set(range(items)) - positive[user]), dtype=np.int64)
        chosen = rng.choice(available, size=min(count, len(available)), replace=False)
        rows.extend([user] * len(chosen)); cols.extend(chosen.tolist())
    rating = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(users, items))
    return -_normalize_bipartite(rating)


def sampled_negative_graph_semantic(split, users, items, seed,
                                    semantic_similarity, exclusion_fraction=0.0):
    """Sample a fixed-size negative graph after excluding top semantic matches.

    Only ``split`` interactions are treated as observed.  The same per-user edge
    budget is used for every exclusion fraction.  If filtering ever leaves too
    few candidates, the least-similar filtered candidates are restored.
    """
    if not 0.0 <= exclusion_fraction < 1.0:
        raise ValueError("exclusion_fraction must be in [0, 1)")
    semantic_similarity = np.asarray(semantic_similarity, dtype=np.float32)
    if semantic_similarity.shape != (users, items):
        raise ValueError(f"Expected semantic matrix {(users, items)}, got {semantic_similarity.shape}")
    rng = np.random.default_rng(seed)
    positive = seen_items(split, users)
    universe = np.arange(items, dtype=np.int64)
    rows, cols = [], []
    filtered_sum = sampled_sum = 0.0
    filtered_count = sampled_count = 0
    insufficient_users = 0
    min_candidates_after_filter = items
    for user in range(users):
        observed = positive[user]
        available = np.asarray([item for item in universe if int(item) not in observed], dtype=np.int64)
        needed = max(1, len(observed))
        filter_count = int(np.floor(exclusion_fraction * len(available)))
        if filter_count:
            similarities = semantic_similarity[user, available]
            local = np.argpartition(similarities, -filter_count)[-filter_count:]
            blocked = available[local]
            keep_mask = np.ones(len(available), dtype=bool)
            keep_mask[local] = False
            eligible = available[keep_mask]
            filtered_sum += float(semantic_similarity[user, blocked].sum(dtype=np.float64))
            filtered_count += len(blocked)
        else:
            blocked = np.empty(0, dtype=np.int64)
            eligible = available
        min_candidates_after_filter = min(min_candidates_after_filter, len(eligible))
        if len(eligible) < needed:
            insufficient_users += 1
            missing = needed - len(eligible)
            restore_order = np.argsort(semantic_similarity[user, blocked])
            eligible = np.concatenate((eligible, blocked[restore_order[:missing]]))
        chosen = rng.choice(eligible, size=needed, replace=False)
        rows.extend([user] * len(chosen)); cols.extend(chosen.tolist())
        sampled_sum += float(semantic_similarity[user, chosen].sum(dtype=np.float64))
        sampled_count += len(chosen)
    rating = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(users, items))
    diagnostics = {
        "edge_count": len(rows),
        "exclusion_fraction": float(exclusion_fraction),
        "filtered_candidate_count": filtered_count,
        "filtered_mean_semantic_similarity": (
            filtered_sum / filtered_count if filtered_count else None),
        "sampled_mean_semantic_similarity": sampled_sum / max(1, sampled_count),
        "users_insufficient_after_filter": insufficient_users,
        "min_candidates_after_filter": int(min_candidates_after_filter),
        "fallback_used": insufficient_users > 0,
        "sampling_seed": int(seed),
        "interaction_source": "training split only",
    }
    return -_normalize_bipartite(rating), diagnostics


def arrays(split):
    return np.asarray(split["users"], dtype=np.int64), np.asarray(split["pos_item"], dtype=np.int64)


def seen_items(train, users: int):
    result = [set() for _ in range(users)]
    us, it = arrays(train)
    for user, item in zip(us, it):
        result[int(user)].add(int(item))
    return result
