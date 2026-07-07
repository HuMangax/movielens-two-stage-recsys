"""Two-tower retrieval model.

User tower: id embedding.
Item tower: id embedding + linear projection of genre features, plus a
learned scalar popularity bias.

Score(u, i) = <user_vec, item_vec> + item_bias_i

The bias is folded into the exported matrices as an extra dimension
(item vectors get their bias appended, user vectors get a constant 1),
so serving-time retrieval over the whole catalog is one matrix multiply.
"""

import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_item_feats: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.item_feat_proj = nn.Linear(n_item_feats, dim, bias=False)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def user_vec(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_emb(users)

    def item_vec(self, items: torch.Tensor, item_feats: torch.Tensor) -> torch.Tensor:
        return self.item_emb(items) + self.item_feat_proj(item_feats[items])

    def score(
        self, users: torch.Tensor, items: torch.Tensor, item_feats: torch.Tensor
    ) -> torch.Tensor:
        u = self.user_vec(users)
        v = self.item_vec(items, item_feats)
        return (u * v).sum(-1) + self.item_bias(items).squeeze(-1)

    @torch.no_grad()
    def export_embeddings(
        self, item_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full user/item matrices with the bias trick applied.

        Returns (users: [n_users, dim+1], items: [n_items, dim+1]) such that
        users @ items.T reproduces ``score`` for every pair.
        """
        n_users = self.user_emb.num_embeddings
        n_items = self.item_emb.num_embeddings
        device = self.user_emb.weight.device
        all_users = torch.arange(n_users, device=device)
        all_items = torch.arange(n_items, device=device)

        u = self.user_vec(all_users)
        v = self.item_vec(all_items, item_feats)
        b = self.item_bias(all_items)
        ones = torch.ones(n_users, 1, device=device)
        return torch.cat([u, ones], dim=1), torch.cat([v, b], dim=1)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking: -log sigmoid(pos - neg), averaged."""
    return -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
