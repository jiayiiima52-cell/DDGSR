from __future__ import annotations

import math
import numpy as np


def evaluate(scores, test, seen, k=10):
    users = np.asarray(test["users"], dtype=np.int64)
    positives = np.asarray(test["pos_item"], dtype=np.int64)
    ranks = []
    for user, positive in zip(users, positives):
        row = scores[user].copy()
        for item in seen[user]:
            if item != positive:
                row[item] = -np.inf
        top = np.argpartition(row, -k)[-k:]
        top = top[np.argsort(row[top])[::-1]]
        found = np.flatnonzero(top == positive)
        ranks.append(int(found[0]) if found.size else None)
    hit = [r for r in ranks if r is not None]
    n = max(1, len(ranks))
    return {
        f"HR@{k}": len(hit) / n,
        f"MRR@{k}": sum(1 / (r + 1) for r in hit) / n,
        f"nDCG@{k}": sum(1 / math.log2(r + 2) for r in hit) / n,
        f"Precision@{k}": len(hit) / (n * k),
    }


def evaluate_grouped(scores, test, seen, k=5):
    """User-level macro metrics matching Eqs. (16)-(21) of TSSGCF."""
    grouped = {}
    for user, item in zip(test["users"], test["pos_item"]):
        grouped.setdefault(int(user), set()).add(int(item))
    recalls, precisions, ndcgs = [], [], []
    for user, relevant in grouped.items():
        row = scores[user].copy()
        for item in seen[user]:
            row[item] = -np.inf
        top = np.argpartition(row, -k)[-k:]
        top = top[np.argsort(row[top])[::-1]]
        hits = [rank for rank, item in enumerate(top) if int(item) in relevant]
        hit_count = len(hits)
        recalls.append(hit_count / len(relevant))
        precisions.append(hit_count / k)
        dcg = sum(1 / math.log2(rank + 2) for rank in hits)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    recall, precision, ndcg = map(float, (np.mean(recalls), np.mean(precisions), np.mean(ndcgs)))
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {f"Recall@{k}": recall, f"nDCG@{k}": ndcg,
            f"Precision@{k}": precision, f"F1@{k}": f1}


def evaluate_by_item_degree(scores, test, seen, train, item_count, k=5):
    train_items = np.asarray(train["pos_item"], dtype=np.int64)
    degrees = np.bincount(train_items, minlength=item_count)
    groups = {
        "Cold": set(np.flatnonzero(degrees == 0)),
        "Tail": set(np.flatnonzero((degrees >= 1) & (degrees <= 2))),
        "Mid": set(np.flatnonzero((degrees >= 3) & (degrees <= 10))),
        "Head": set(np.flatnonzero(degrees > 10)),
    }
    grouped_test = {}
    for user, item in zip(test["users"], test["pos_item"]):
        grouped_test.setdefault(int(user), set()).add(int(item))
    output = {}
    for name, members in groups.items():
        recalls, precisions, ndcgs = [], [], []
        relevant_interactions = 0
        for user, all_relevant in grouped_test.items():
            relevant = all_relevant & members
            if not relevant:
                continue
            row = scores[user].copy()
            for item in seen[user]: row[item] = -np.inf
            top = np.argpartition(row, -k)[-k:]
            top = top[np.argsort(row[top])[::-1]]
            hits = [rank for rank, item in enumerate(top) if int(item) in relevant]
            recalls.append(len(hits) / len(relevant)); precisions.append(len(hits) / k)
            dcg = sum(1 / math.log2(rank + 2) for rank in hits)
            ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
            ndcgs.append(dcg / ideal if ideal else 0.0)
            relevant_interactions += len(relevant)
        recall = float(np.mean(recalls)) if recalls else 0.0
        precision = float(np.mean(precisions)) if precisions else 0.0
        ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        output[name] = {f"Recall@{k}": recall, f"nDCG@{k}": ndcg,
                        f"Precision@{k}": precision, f"F1@{k}": f1,
                        "users": len(recalls), "test_interactions": relevant_interactions,
                        "items": len(members)}
    return output
