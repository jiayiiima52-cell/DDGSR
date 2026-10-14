from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config


def scipy_to_torch(matrix, device):
    coo = matrix.tocoo()
    indices = torch.from_numpy(__import__("numpy").vstack((coo.row, coo.col))).long().to(device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, coo.shape, device=device, check_invariants=False
    ).coalesce()


class DDGSR(nn.Module):
    """DDGSR dual-graph encoder with semantic residual scoring."""

    def __init__(self, config: Config, text_features=None, degree_gates=None,
                 global_features=None, auxiliary_weight=0.01, fixed_text=False,
                 semantic_residual=False, semantic_score_weight=0.1,
                 conflict_semantic_gate=False, conflict_gate_beta=1.0,
                 constant_semantic_gate=False, disagreement_only_gate=False,
                 view_mode="dual", disagreement_residual_gain=None,
                 semantic_score_features=None, view_fusion_weight=None):
        super().__init__()
        self.cfg = config
        self.pos_user = nn.Embedding(config.users, config.dim)
        self.pos_item = nn.Embedding(config.items, config.dim)
        self.neg_user = nn.Embedding(config.users, config.dim)
        self.neg_item = nn.Embedding(config.items, config.dim)
        self.use_semantic = text_features is not None
        self.use_global = global_features is not None
        self.use_semantic_score = semantic_score_features is not None
        self.auxiliary_weight = auxiliary_weight
        self.current_auxiliary_weight = auxiliary_weight
        self.fixed_text = fixed_text
        self.semantic_residual = semantic_residual
        self.semantic_score_weight = semantic_score_weight
        self.conflict_semantic_gate = conflict_semantic_gate
        self.conflict_gate_beta = conflict_gate_beta
        self.constant_semantic_gate = constant_semantic_gate
        self.disagreement_only_gate = disagreement_only_gate
        self.disagreement_residual_gain = disagreement_residual_gain
        if view_fusion_weight is not None and not 0.0 <= view_fusion_weight <= 1.0:
            raise ValueError("view_fusion_weight must be in [0, 1]")
        self.view_fusion_weight = view_fusion_weight
        if view_mode not in {"dual", "interest", "disinterest"}:
            raise ValueError(f"Unknown DDGSR view mode: {view_mode}")
        self.view_mode = view_mode
        if self.use_semantic:
            user_text, item_text = text_features
            self.register_buffer("user_text", user_text.float())
            self.register_buffer("item_text", item_text.float())
            self.register_buffer("user_gate", degree_gates[0].float().unsqueeze(1))
            self.register_buffer("item_gate", degree_gates[1].float().unsqueeze(1))
            if not fixed_text:
                self.user_text_proj = nn.Linear(user_text.shape[1], config.dim, bias=False)
                self.item_text_proj = nn.Linear(item_text.shape[1], config.dim, bias=False)
        if self.use_global:
            self.register_buffer("global_user", global_features[0].float())
            self.register_buffer("global_item", global_features[1].float())
        if self.use_semantic_score:
            self.register_buffer("score_user_text", semantic_score_features[0].float())
            self.register_buffer("score_item_text", semantic_score_features[1].float())
        # A local stream makes every embedding depend on the experiment seed.
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        for embedding in (self.pos_user, self.pos_item, self.neg_user, self.neg_item):
            with torch.no_grad():
                embedding.weight.copy_(torch.randn(embedding.weight.shape, generator=generator) * 0.05)

    @staticmethod
    def propagate(adjacency, initial, layers):
        outputs = [initial]
        hidden = initial
        for _ in range(layers):
            hidden = torch.sparse.mm(adjacency, hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=0).mean(dim=0)

    def embeddings(self, interest_graph, disinterest_graph):
        positive = torch.cat((self.pos_user.weight, self.pos_item.weight), dim=0)
        negative = torch.cat((self.neg_user.weight, self.neg_item.weight), dim=0)
        hp = self.propagate(interest_graph, positive, self.cfg.layers)
        hn = self.propagate(disinterest_graph, negative, self.cfg.neg_layers)
        pu, pi, nu, ni = hp[:self.cfg.users], hp[self.cfg.users:], hn[:self.cfg.users], hn[self.cfg.users:]
        if self.use_semantic:
            tu = F.normalize(self.user_text if self.fixed_text else self.user_text_proj(self.user_text), dim=1)
            ti = F.normalize(self.item_text if self.fixed_text else self.item_text_proj(self.item_text), dim=1)
            pu_scale = pu.norm(dim=1, keepdim=True).mean().detach().clamp_min(1e-3)
            pi_scale = pi.norm(dim=1, keepdim=True).mean().detach().clamp_min(1e-3)
            pu = self.user_gate * pu + (1 - self.user_gate) * tu * pu_scale
            pi = self.item_gate * pi + (1 - self.item_gate) * ti * pi_scale
        return pu, pi, nu, ni

    @staticmethod
    def bpr(positive, negative, weights=None):
        losses = -F.logsigmoid(positive - negative)
        if weights is not None:
            losses = losses * weights
        return losses.mean()

    def semantic_pair_gate(self, users, items, interest_scores=None, disinterest_scores=None):
        if not self.conflict_semantic_gate:
            return (1 - self.user_gate[users].squeeze(1) if self.use_semantic
                    else torch.ones_like(interest_scores))
        if self.constant_semantic_gate:
            return torch.ones_like(interest_scores)
        disagreement = torch.abs(torch.sigmoid(interest_scores) - torch.sigmoid(disinterest_scores))
        if self.disagreement_only_gate:
            disagreement_gate = 1 - (1 - disagreement).pow(self.conflict_gate_beta)
            if self.disagreement_residual_gain is not None:
                return 1 + self.disagreement_residual_gain * disagreement_gate
            return disagreement_gate
        user_uncertainty = 1 - self.user_gate[users].squeeze(1)
        item_uncertainty = 1 - self.item_gate[items].squeeze(1)
        degree_uncertainty = torch.sqrt(user_uncertainty * item_uncertainty)
        # Probabilistic OR: either sparse structural evidence or conflicting
        # interest/disinterest views is sufficient to invoke semantic evidence.
        return 1 - (1 - degree_uncertainty) * (1 - disagreement).pow(self.conflict_gate_beta)

    def semantic_pair_score(self, users, items, interest_scores=None, disinterest_scores=None):
        tu = F.normalize(self.score_user_text if self.use_semantic_score else self.user_text, dim=1)
        ti = F.normalize(self.score_item_text if self.use_semantic_score else self.item_text, dim=1)
        gate = self.semantic_pair_gate(users, items, interest_scores, disinterest_scores)
        return self.semantic_score_weight * gate * (tu[users] * ti[items]).sum(1)

    def contrastive(self, view1, view2):
        a = F.normalize(view1, dim=1)
        b = F.normalize(view2, dim=1)
        logits = a @ b.T / self.cfg.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def loss(self, users, positives, negatives, graphs, rank_weights=None):
        pu, pi, nu, ni = self.embeddings(*graphs)
        pos_interest = (pu[users] * pi[positives]).sum(1)
        neg_interest = (pu[users] * pi[negatives]).sum(1)
        pos_disinterest = (nu[users] * ni[positives]).sum(1)
        neg_disinterest = (nu[users] * ni[negatives]).sum(1)
        if self.view_mode == "interest":
            pos_score, neg_score = pos_interest, neg_interest
        elif self.view_mode == "disinterest":
            pos_score, neg_score = pos_disinterest, neg_disinterest
        else:
            if self.view_fusion_weight is None:
                pos_score = pos_interest + pos_disinterest
                neg_score = neg_interest + neg_disinterest
            else:
                eta = self.view_fusion_weight
                pos_score = eta * pos_interest + (1 - eta) * pos_disinterest
                neg_score = eta * neg_interest + (1 - eta) * neg_disinterest
        if self.semantic_residual:
            pos_score = pos_score + self.semantic_pair_score(
                users, positives, pos_interest, pos_disinterest)
            neg_score = neg_score + self.semantic_pair_score(
                users, negatives, neg_interest, neg_disinterest)
        bpr_loss = self.bpr(pos_score, neg_score, rank_weights)
        # Cross-view alignment is defined only when both DDGSR views are present.
        cl_loss = (self.contrastive(pu, nu) + self.contrastive(pi, ni)
                   if self.view_mode == "dual" else torch.zeros((), device=users.device))
        auxiliary = torch.zeros((), device=users.device)
        if self.use_global:
            auxiliary = (1 - F.cosine_similarity(pu[users], self.global_user[users], dim=1)).mean()
            auxiliary = auxiliary + (1 - F.cosine_similarity(pi[positives], self.global_item[positives], dim=1)).mean()
        return bpr_loss + self.cfg.cl_weight * cl_loss + self.current_auxiliary_weight * auxiliary, bpr_loss.detach(), cl_loss.detach()

    @torch.no_grad()
    def scores(self, graphs):
        pu, pi, nu, ni = self.embeddings(*graphs)
        interest = pu @ pi.T
        disinterest = nu @ ni.T
        if self.view_mode == "interest":
            result = interest
        elif self.view_mode == "disinterest":
            result = disinterest
        else:
            if self.view_fusion_weight is None:
                result = interest + disinterest
            else:
                eta = self.view_fusion_weight
                result = eta * interest + (1 - eta) * disinterest
        if self.semantic_residual:
            semantic = (F.normalize(self.score_user_text if self.use_semantic_score else self.user_text, dim=1) @
                        F.normalize(self.score_item_text if self.use_semantic_score else self.item_text, dim=1).T)
            if self.conflict_semantic_gate:
                if self.constant_semantic_gate:
                    gate = torch.ones_like(interest)
                else:
                    disagreement = torch.abs(torch.sigmoid(interest) - torch.sigmoid(disinterest))
                    if self.disagreement_only_gate:
                        disagreement_gate = 1 - (1 - disagreement).pow(self.conflict_gate_beta)
                        if self.disagreement_residual_gain is not None:
                            gate = 1 + self.disagreement_residual_gain * disagreement_gate
                        else:
                            gate = disagreement_gate
                    else:
                        user_uncertainty = 1 - self.user_gate
                        item_uncertainty = (1 - self.item_gate).T
                        degree_uncertainty = torch.sqrt(user_uncertainty * item_uncertainty)
                        gate = 1 - (1 - degree_uncertainty) * (1 - disagreement).pow(
                            self.conflict_gate_beta)
            else:
                gate = 1 - self.user_gate
            result = result + self.semantic_score_weight * gate * semantic
        return result

    @torch.no_grad()
    def semantic_gate_matrix(self, graphs):
        if not self.conflict_semantic_gate:
            return None
        pu, pi, nu, ni = self.embeddings(*graphs)
        interest, disinterest = pu @ pi.T, nu @ ni.T
        if self.constant_semantic_gate:
            return torch.ones_like(interest)
        disagreement = torch.abs(torch.sigmoid(interest) - torch.sigmoid(disinterest))
        if self.disagreement_only_gate:
            disagreement_gate = 1 - (1 - disagreement).pow(self.conflict_gate_beta)
            if self.disagreement_residual_gain is not None:
                return 1 + self.disagreement_residual_gain * disagreement_gate
            return disagreement_gate
        degree_uncertainty = torch.sqrt((1 - self.user_gate) * (1 - self.item_gate).T)
        return 1 - (1 - degree_uncertainty) * (1 - disagreement).pow(
            self.conflict_gate_beta)
