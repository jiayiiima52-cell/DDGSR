from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiment_config import GRAPH_SEED, seed_everything
from ddgsrl.config import Config
from ddgsrl.data import arrays, load_dataset, seen_items, sampled_negative_graph_semantic
from ddgsrl.graph import update_graphs
from ddgsrl.metrics import evaluate, evaluate_grouped, evaluate_by_item_degree
from ddgsrl.model import DDGSRL, scipy_to_torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate DDGSRL")
    parser.add_argument("--dataset", choices=["ml-100k", "gowalla", "citeulike", "programmableweb", "tssgcf"], default="ml-100k")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5, help="Recommendation cutoff used for evaluation")
    parser.add_argument("--full-train", action="store_true", help="Use all official training interactions")
    parser.add_argument("--selection-only", action="store_true",
                        help="Validation-only run: do not load or evaluate the test split")
    parser.add_argument("--variant", choices=["base", "semantic", "semantic_ns", "semantic_svd", "stable_svd", "rank_rerank", "rank_loss", "rank_full", "conflict_gate", "ddgsrl"], default="base")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs",
                        help="Directory for result/checkpoint artifacts")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Write the result JSON only; skip checkpoint and graph artifacts")
    parser.add_argument("--semantic-score-weight", type=float, default=0.1)
    parser.add_argument("--conflict-gate-beta", type=float, default=1.0,
                        help="Strength of dual-graph disagreement in the semantic gate")
    parser.add_argument("--constant-semantic-gate", action="store_true",
                        help="Use gate=1 as the ungated semantic-residual control")
    parser.add_argument("--disagreement-only-gate", action="store_true",
                        help="Ablation: remove degree uncertainty and gate only by dual-graph disagreement")
    parser.add_argument("--disagreement-residual-gain", type=float, default=None,
                        help="Preserve SR and use DD as additive modulation: gate=1+alpha*DD")
    parser.add_argument("--view-fusion-weight", type=float, default=None,
                        help="Interest-view weight eta in eta*s_pos+(1-eta)*s_neg")
    parser.add_argument("--ddgsrl-view", choices=["dual", "interest", "disinterest"], default="dual",
                        help="DDGSRL view ablation: retain both, interest only, or disinterest only")
    parser.add_argument("--no-contrastive", action="store_true",
                        help="DDGSRL ablation: set the cross-view contrastive-loss weight to zero")
    parser.add_argument("--no-graph-update", action="store_true", help="Disable Eqs. (18)-(24) for ablation")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--neg-layers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float,
                        help="L2 regularization applied by the optimizer")
    parser.add_argument("--cl-weight", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--graph-ratio", type=float)
    parser.add_argument("--graph-weight", type=float)
    parser.add_argument("--update-every", type=int)
    parser.add_argument("--eval-ks", default="",
                        help="Optional comma-separated additional K values evaluated from the same final scores")
    parser.add_argument("--hard-negative-sampling", action="store_true",
                        help="Sample high-scoring unseen negatives after the ranking warmup")
    parser.add_argument("--ndcg-weighted-bpr", action="store_true",
                        help="Weight BPR pairs by their estimated NDCG@K displacement")
    parser.add_argument("--hard-candidate-count", type=int, default=32)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--ndcg-weight-strength", type=float, default=3.0)
    parser.add_argument("--rank-warmup-epochs", type=int, default=3)
    parser.add_argument("--selection-patience", type=int, default=None,
                        help="Validation evaluations without improvement before early stopping")
    parser.add_argument("--negative-semantic-exclusion", type=float, default=None,
                        help="Rebuild the initial negative graph from training interactions after excluding this top semantic fraction")
    parser.add_argument("--negative-graph-seed", type=int, default=GRAPH_SEED)
    return parser.parse_args()


def negatives(users, positives, item_count, seen, rng, semantic_scores=None, threshold=0.34,
              model_scores=None, candidate_count=32, hard_ratio=1.0):
    result = rng.integers(0, item_count, size=len(users), dtype=np.int64)
    for index, user in enumerate(users):
        accepted = False
        candidates = []
        attempts = 0
        use_hard = model_scores is not None and rng.random() < hard_ratio
        target = candidate_count if use_hard else 1
        while len(candidates) < target and attempts < max(100, target * 10):
            attempts += 1
            candidate = int(rng.integers(0, item_count))
            valid = candidate not in seen[int(user)] and candidate != positives[index]
            if valid and (semantic_scores is None or semantic_scores[int(user), candidate] < threshold):
                candidates.append(candidate)
        if candidates:
            result[index] = (max(candidates, key=lambda item: model_scores[int(user), item])
                             if use_hard else candidates[0])
            accepted = True
        if not accepted:
            available = [item for item in range(item_count)
                         if item not in seen[int(user)] and item != positives[index]]
            result[index] = int(rng.choice(available))
    return result


def ndcg_pair_weights(scores, users, positives, negatives, seen, k=5, strength=5.0):
    weights = np.ones(len(users), dtype=np.float32)
    discounts = np.asarray([1 / np.log2(rank + 2) for rank in range(k)], dtype=np.float32)
    for index, (user, positive, negative) in enumerate(zip(users, positives, negatives)):
        row = scores[int(user)].copy()
        # Other observed training positives are not candidates at evaluation time.
        # Masking them prevents their high scores from distorting the sampled
        # positive/negative rank displacement used by the top-k surrogate.
        masked = seen[int(user)] - {int(positive)}
        if masked:
            row[np.fromiter(masked, dtype=np.int64)] = -np.inf
        pos_rank = int(np.sum(row > row[int(positive)]))
        neg_rank = int(np.sum(row > row[int(negative)]))
        pos_gain = discounts[pos_rank] if pos_rank < k else 0.0
        neg_gain = discounts[neg_rank] if neg_rank < k else 0.0
        weights[index] += strength * abs(float(pos_gain - neg_gain))
    return weights


def load_text_features(folder, device):
    return (torch.load(folder / "mashup_descr_emb.pt", map_location=device, weights_only=True),
            torch.load(folder / "api_descr_emb.pt", map_location=device, weights_only=True))


def deterministic_pca_text(text_features, dim, device, cache_path=None):
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        return (torch.from_numpy(cached["user"]).to(device),
                torch.from_numpy(cached["item"]).to(device))
    user, item = (tensor.detach().cpu().numpy().astype(np.float64) for tensor in text_features)
    joined = np.concatenate((user, item), axis=0)
    joined -= joined.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(joined, full_matrices=False)
    projected = (joined @ vt[:dim].T).astype(np.float32)
    projected_user, projected_item = projected[:len(user)], projected[len(user):]
    if cache_path is not None:
        np.savez_compressed(cache_path, user=projected_user, item=projected_item)
    return (torch.from_numpy(projected_user).to(device),
            torch.from_numpy(projected_item).to(device))


def degree_gates(train, users, items, device):
    us, it = arrays(train)
    ud = torch.from_numpy(np.bincount(us, minlength=users)).float().to(device)
    idg = torch.from_numpy(np.bincount(it, minlength=items)).float().to(device)
    return ud / (ud + 2.0), idg / (idg + 2.0)


def svd_features(train, users, items, dim, device):
    us, it = arrays(train)
    matrix = sp.csr_matrix((np.ones(len(us)), (us, it)), shape=(users, items), dtype=np.float32)
    u, s, vt = svds(matrix, k=min(dim, min(users, items) - 1), random_state=100)
    order = np.argsort(s)[::-1]; u, s, vt = u[:, order], s[order], vt[order]
    root = np.sqrt(s)
    return (torch.from_numpy((u * root).astype(np.float32)).to(device),
            torch.from_numpy((vt.T * root).astype(np.float32)).to(device))


def main():
    args = parse_args()
    if args.hard_candidate_count < 1:
        raise ValueError("--hard-candidate-count must be positive")
    if not 0.0 <= args.hard_negative_ratio <= 1.0:
        raise ValueError("--hard-negative-ratio must be in [0, 1]")
    if args.ndcg_weight_strength < 0 or args.rank_warmup_epochs < 0:
        raise ValueError("ranking weight and warmup must be non-negative")
    if args.selection_patience is not None and args.selection_patience < 1:
        raise ValueError("selection patience must be positive")
    if args.constant_semantic_gate and args.disagreement_only_gate:
        raise ValueError("--constant-semantic-gate and --disagreement-only-gate are mutually exclusive")
    if args.ddgsrl_view != "dual" and args.variant not in {"base", "ddgsrl"}:
        raise ValueError("Single-view DDGSRL ablation is supported only for base or ddgsrl")
    if args.disagreement_residual_gain is not None:
        if args.variant not in {"conflict_gate", "ddgsrl"} or not args.disagreement_only_gate:
            raise ValueError("Residual-preserving DD requires a disagreement-only gate")
        if args.disagreement_residual_gain < 0:
            raise ValueError("Disagreement residual gain must be non-negative")
    overrides = {}
    if args.epochs: overrides["epochs"] = args.epochs
    if args.seed is not None: overrides["seed"] = args.seed
    if args.no_contrastive: overrides["cl_weight"] = 0.0
    for key in ["dim", "layers", "neg_layers", "batch_size", "learning_rate", "weight_decay",
                "cl_weight", "temperature", "graph_ratio", "graph_weight", "update_every"]:
        value = getattr(args, key)
        if value is not None: overrides[key] = value
    config = Config.for_dataset(args.dataset, **overrides)
    seed_everything(config.seed)
    device = torch.device(args.device)
    started_at = time.time()
    if args.selection_only and args.full_train:
        raise ValueError("--selection-only and --full-train cannot be used together")
    train, val, test, original_pos, original_neg, folder = load_dataset(
        args.data_root, args.dataset, full_train=args.full_train,
        load_test=not args.selection_only)
    train_users, train_items = arrays(train)
    seen = seen_items(train, config.users)
    seen_eval = [set(items) for items in seen]
    if test is not None:
        val_seen_users, val_seen_items = arrays(val)
        for user, item in zip(val_seen_users, val_seen_items):
            seen_eval[int(user)].add(int(item))
    rng = np.random.default_rng(config.seed)
    negative_graph_diagnostics = None
    if args.negative_semantic_exclusion is not None:
        raw_user_text, raw_item_text = load_text_features(folder, device)
        with torch.no_grad():
            semantic_for_graph = (
                torch.nn.functional.normalize(raw_user_text, dim=1) @
                torch.nn.functional.normalize(raw_item_text, dim=1).T
            ).cpu().numpy()
        original_neg, negative_graph_diagnostics = sampled_negative_graph_semantic(
            train, config.users, config.items, args.negative_graph_seed,
            semantic_for_graph, args.negative_semantic_exclusion)
    clean = args.variant == "ddgsrl"
    semantic = args.variant != "base"
    text_features = load_text_features(folder, device) if semantic and not clean else None
    semantic_score_features = None
    rank_variants = {"rank_rerank", "rank_loss", "rank_full", "conflict_gate"}
    fixed_text = args.variant == "stable_svd" or args.variant in rank_variants
    if fixed_text:
        text_features = deterministic_pca_text(
            text_features, config.dim, device, folder / f"text_pca_{config.dim}.npz")
    if clean:
        semantic_score_features = deterministic_pca_text(
            load_text_features(folder, device), config.dim, device,
            folder / f"text_pca_{config.dim}.npz")
    gates = degree_gates(train, config.users, config.items, device) if semantic and not clean else None
    global_features = svd_features(train, config.users, config.items, config.dim, device) if args.variant in {"semantic_svd", "stable_svd", *rank_variants} else None
    semantic_scores = None
    if args.variant in {"semantic_ns", "semantic_svd", "stable_svd", *rank_variants}:
        raw_text = load_text_features(folder, device) if fixed_text else text_features
        semantic_scores = (torch.nn.functional.normalize(raw_text[0], dim=1) @
                           torch.nn.functional.normalize(raw_text[1], dim=1).T).cpu().numpy()
    model = DDGSRL(config, text_features=text_features, degree_gates=gates,
                 global_features=global_features, fixed_text=fixed_text,
                 semantic_residual=args.variant in rank_variants or clean,
                 semantic_score_weight=args.semantic_score_weight,
                 conflict_semantic_gate=args.variant in {"conflict_gate", "ddgsrl"},
                 conflict_gate_beta=args.conflict_gate_beta,
                 constant_semantic_gate=args.constant_semantic_gate,
                 disagreement_only_gate=args.disagreement_only_gate,
                 view_mode=args.ddgsrl_view,
                 disagreement_residual_gain=args.disagreement_residual_gain,
                 semantic_score_features=semantic_score_features,
                 view_fusion_weight=args.view_fusion_weight).to(device)
    optimizer = torch.optim.NAdam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    current_pos, current_neg = original_pos, original_neg
    best_loss, best_score, stale, best_state, best_epoch, last_state = float("inf"), -float("inf"), 0, None, 0, None
    best_graph_state, best_validation_metrics = None, None
    average_interactions = len(train_users) / config.users

    test_size = "not-loaded" if test is None else str(len(test["users"]))
    print(f"Dataset: {folder}\nDevice: {device}\nTrain/val/test: "
          f"{len(train_users)}/{len(val['users'])}/{test_size}")
    training_started_at = time.perf_counter()
    for epoch in range(config.epochs):
        if args.variant == "stable_svd" or args.variant in rank_variants:
            model.current_auxiliary_weight = 0.01 * min(1.0, max(0.0, (epoch + 1 - 5) / 10))
        model.train()
        graphs = (scipy_to_torch(current_pos, device), scipy_to_torch(current_neg, device))
        epoch_scores = None
        ranking_active = epoch + 1 > args.rank_warmup_epochs
        if (args.variant in {"rank_loss", "rank_full", "conflict_gate"}
                or ranking_active and (args.hard_negative_sampling or args.ndcg_weighted_bpr)):
            model.eval()
            with torch.no_grad(): epoch_scores = model.scores(graphs).cpu().numpy()
            model.train()
        legacy_hard = args.variant in {"rank_full", "conflict_gate"} and epoch >= 5
        hard_scores = epoch_scores if legacy_hard or (ranking_active and args.hard_negative_sampling) else None
        sampled = negatives(train_users, train_items, config.items, seen, rng, semantic_scores,
                            model_scores=hard_scores,
                            candidate_count=args.hard_candidate_count,
                            hard_ratio=(args.hard_negative_ratio if args.hard_negative_sampling else 1.0))
        rank_weights = (ndcg_pair_weights(epoch_scores, train_users, train_items, sampled, seen, k=args.top_k)
                        if args.variant in {"rank_loss", "rank_full", "conflict_gate"}
                        else ndcg_pair_weights(
                            epoch_scores, train_users, train_items, sampled, seen, k=args.top_k,
                            strength=args.ndcg_weight_strength)
                        if ranking_active and args.ndcg_weighted_bpr
                        else np.ones(len(train_users), dtype=np.float32))
        dataset = TensorDataset(*(torch.from_numpy(x) for x in (train_users, train_items, sampled, rank_weights)))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        total = 0.0
        for user, positive, negative, weight in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            user, positive, negative, weight = user.to(device), positive.to(device), negative.to(device), weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model.loss(user, positive, negative, graphs, weight)
            loss.backward(); optimizer.step(); total += loss.item() * len(user)
        train_loss = total / len(dataset)

        model.eval()
        with torch.no_grad():
            val_users, val_items = arrays(val)
            val_neg = negatives(val_users, val_items, config.items, seen, rng, semantic_scores)
            val_loss, _, _ = model.loss(
                torch.from_numpy(val_users).to(device), torch.from_numpy(val_items).to(device),
                torch.from_numpy(val_neg).to(device), graphs)
        selection_value = -val_loss.item()
        val_ndcg = None
        val_metrics = None
        if (args.variant == "stable_svd" or args.variant in rank_variants or clean) and not args.full_train:
            with torch.no_grad():
                val_scores = model.scores(graphs).cpu().numpy()
            val_metrics = evaluate_grouped(val_scores, val, seen, k=args.top_k)
            if args.eval_ks:
                for extra_k in sorted({int(value) for value in args.eval_ks.split(",") if value.strip()}):
                    if extra_k != args.top_k:
                        val_metrics.update(evaluate_grouped(val_scores, val, seen, k=extra_k))
            val_ndcg = val_metrics[f"nDCG@{args.top_k}"]
            selection_value = val_ndcg
        extra = f" val_ndcg={val_ndcg:.6f}" if val_ndcg is not None else ""
        print(f"epoch={epoch + 1:04d} train_loss={train_loss:.6f} val_loss={val_loss.item():.6f}{extra}")
        last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if selection_value > best_score:
            best_score, best_loss = selection_value, val_loss.item()
            stale, best_epoch = 0, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_graph_state = (current_pos.copy(), current_neg.copy())
            best_validation_metrics = val_metrics
        else:
            stale += 1
        selection_patience = (args.selection_patience if args.selection_patience is not None
                              else (10 if args.variant in rank_variants or clean else config.patience))
        should_stop = (stale >= selection_patience
                       and epoch + 1 >= (20 if args.variant in rank_variants or clean else 0)
                       and not args.full_train)

        if not args.no_graph_update and (epoch + 1) % config.update_every == 0:
            with torch.no_grad():
                embeddings = model.embeddings(*graphs)
            current_pos, current_neg = update_graphs(
                original_pos, original_neg, embeddings, config.users, config.items,
                config.graph_ratio, config.graph_weight, average_interactions,
                view_mode=args.ddgsrl_view)
        if should_stop:
            print("Early stopping")
            break
    training_elapsed_seconds = time.perf_counter() - training_started_at

    if args.full_train:
        best_state = last_state
        best_graph_state = (current_pos.copy(), current_neg.copy())
    model.load_state_dict(best_state)
    model.eval()
    evaluation_pos, evaluation_neg = best_graph_state
    graphs = (scipy_to_torch(evaluation_pos, device), scipy_to_torch(evaluation_neg, device))
    result, subgroup, multi_k_metrics = None, None, None
    inference_seconds = None
    if not args.selection_only:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started_at = time.perf_counter()
        scores = model.scores(graphs).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started_at
        result = (evaluate_grouped(scores, test, seen_eval, k=args.top_k)
                  if args.dataset == "tssgcf" else evaluate(scores, test, seen_eval, k=args.top_k))
        if args.eval_ks:
            ks = sorted({int(value) for value in args.eval_ks.split(",") if value.strip()})
            multi_k_metrics = {
                str(k): (evaluate_grouped(scores, test, seen_eval, k=k)
                         if args.dataset == "tssgcf" else evaluate(scores, test, seen_eval, k=k))
                for k in ks
            }
        subgroup = (evaluate_by_item_degree(scores, test, seen_eval, train, config.items, k=args.top_k)
                    if args.dataset == "tssgcf" and args.full_train else None)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    suffix = "" if args.dataset == "ml-100k" else f"_{args.dataset}"
    if args.variant != "base": suffix += f"_{args.variant}"
    if args.run_name: suffix += f"_{args.run_name}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"results{suffix}.json"
    environment = {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "epochs_completed": epoch + 1,
        "best_validation_loss": best_loss,
        "best_selection_score": best_score,
        "best_epoch": best_epoch,
        "top_k": args.top_k,
        "full_train": args.full_train,
        "selection_only": args.selection_only,
        "variant": args.variant,
        "semantic_score_weight": args.semantic_score_weight,
        "conflict_gate_beta": args.conflict_gate_beta,
        "constant_semantic_gate": args.constant_semantic_gate,
        "disagreement_only_gate": args.disagreement_only_gate,
        "disagreement_residual_gain": args.disagreement_residual_gain,
        "view_fusion_weight": args.view_fusion_weight,
        "ddgsrl_view": args.ddgsrl_view,
        "no_contrastive": args.no_contrastive,
        "no_graph_update": args.no_graph_update,
        "hard_negative_sampling": args.hard_negative_sampling,
        "ndcg_weighted_bpr": args.ndcg_weighted_bpr,
        "hard_candidate_count": args.hard_candidate_count,
        "hard_negative_ratio": args.hard_negative_ratio,
        "ndcg_weight_strength": args.ndcg_weight_strength,
        "rank_warmup_epochs": args.rank_warmup_epochs,
        "selection_patience": args.selection_patience,
        "negative_semantic_exclusion": args.negative_semantic_exclusion,
        "negative_graph_seed": args.negative_graph_seed,
        "negative_graph_diagnostics": negative_graph_diagnostics,
        "ddgsrl": clean,
        "metrics_only": args.metrics_only,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "training_elapsed_seconds": training_elapsed_seconds,
        "average_epoch_seconds": training_elapsed_seconds / (epoch + 1),
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                               if device.type == "cuda" else None),
    }
    payload = {"config": vars(config), "environment": environment,
               "validation_metrics": best_validation_metrics, "metrics": result}
    if multi_k_metrics is not None: payload["multi_k_metrics"] = multi_k_metrics
    if subgroup is not None: payload["subgroups"] = subgroup
    if args.variant == "conflict_gate":
        gate = model.semantic_gate_matrix(graphs).cpu().numpy()
        payload["gate_diagnostics"] = {
            "mean": float(gate.mean()), "std": float(gate.std()),
            "p10": float(np.quantile(gate, 0.10)),
            "median": float(np.quantile(gate, 0.50)),
            "p90": float(np.quantile(gate, 0.90)),
        }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.metrics_only:
        torch.save(best_state, args.output_dir / f"ddgsrl_best{suffix}.pt")
        sp.save_npz(args.output_dir / f"ddgsrl_best_interest{suffix}.npz", evaluation_pos)
        sp.save_npz(args.output_dir / f"ddgsrl_best_disinterest{suffix}.npz", evaluation_neg)


if __name__ == "__main__":
    main()
