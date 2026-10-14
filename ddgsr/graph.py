from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _normalize_bipartite(rating):
    users, items = rating.shape
    adjacency = sp.bmat([[None, rating], [rating.T, None]], format="csr", dtype=np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv[nonzero] = degree[nonzero] ** -0.5
    scale = sp.diags(inv)
    return (scale @ adjacency @ scale).tocsr()


def update_graphs(original_interest, original_disinterest, embeddings, users, items,
                  ratio, weight, average_interactions, view_mode="dual"):
    """Dual latent graph learning from Eqs. (18)-(24), always based on originals."""
    pu, pi, nu, ni = (x.detach().cpu().numpy() for x in embeddings)
    if view_mode == "interest":
        user_repr, item_repr = pu, pi
    elif view_mode == "disinterest":
        user_repr, item_repr = nu, ni
    elif view_mode == "dual":
        user_repr = np.concatenate((pu, nu), axis=1)
        item_repr = np.concatenate((pi, ni), axis=1)
    else:
        raise ValueError(f"Unknown DDGSR view mode: {view_mode}")
    user_repr /= np.maximum(np.linalg.norm(user_repr, axis=1, keepdims=True), 1e-12)
    item_repr /= np.maximum(np.linalg.norm(item_repr, axis=1, keepdims=True), 1e-12)
    similarity = user_repr @ item_repr.T
    k = max(1, min(items, int(average_interactions * ratio)))
    most = np.argpartition(similarity, -k, axis=1)[:, -k:]
    least = np.argpartition(similarity, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(users), k)
    pos_rating = sp.csr_matrix((np.ones(rows.size), (rows, most.ravel())), shape=(users, items))
    neg_rating = sp.csr_matrix((np.ones(rows.size), (rows, least.ravel())), shape=(users, items))
    latent_pos = _normalize_bipartite(pos_rating)
    latent_neg = -_normalize_bipartite(neg_rating)

    pos_mask = sp.bmat([[None, neg_rating], [neg_rating.T, None]], format="csr")
    neg_mask = sp.bmat([[None, pos_rating], [pos_rating.T, None]], format="csr")
    kept_pos = original_interest - original_interest.multiply(pos_mask.astype(bool))
    kept_neg = original_disinterest - original_disinterest.multiply(neg_mask.astype(bool))
    return ((1 - weight) * kept_pos + weight * latent_pos).tocsr(), \
           ((1 - weight) * kept_neg + weight * latent_neg).tocsr()

