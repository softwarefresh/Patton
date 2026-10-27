"""Small evidence-set scorer over frozen cached vectors; not GraphFormers.

Both modes keep the supplied shared projection as a frozen mean-score base.
An equally sized trainable head scores query/context interactions. Attention
uses a query-dependent context; the mean control uses uniform real evidence.
The residual starts at exactly zero and is bounded by ``residual_scale``.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from partner_evidence_dataset import require


class EvidenceInteractionTeacher(nn.Module):
    def __init__(self, base_weight, hidden_dim=128, mlp_hidden=64, mode="attention",
                 attention_temperature=0.2, residual_scale=0.25):
        super().__init__()
        require(isinstance(base_weight, torch.Tensor) and base_weight.ndim == 2 and
                base_weight.shape[0] == base_weight.shape[1] and base_weight.shape[0] > 0,
                "Base projection must be a nonempty square matrix")
        require(torch.is_floating_point(base_weight) and bool(torch.isfinite(base_weight).all()),
                "Base projection must contain finite floating-point weights")
        require(isinstance(hidden_dim, int) and not isinstance(hidden_dim, bool) and hidden_dim > 0 and
                isinstance(mlp_hidden, int) and not isinstance(mlp_hidden, bool) and mlp_hidden > 0,
                "Head dimensions must be positive integers")
        require(mode in ("attention", "mean"), "Mode must be attention or mean")
        require(math.isfinite(attention_temperature) and attention_temperature > 0,
                "Attention temperature must be positive and finite")
        require(math.isfinite(residual_scale) and residual_scale > 0,
                "Residual scale must be positive and finite")
        self.input_dim = base_weight.shape[1]
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.attention_temperature = float(attention_temperature)
        self.residual_scale = float(residual_scale)
        self.register_buffer("base_weight", base_weight.detach().clone())
        self.head = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(4 * hidden_dim, mlp_hidden), nn.GELU(),
                                 nn.Linear(mlp_hidden, 1))
        self.head.to(device=base_weight.device, dtype=base_weight.dtype)
        self.mlp.to(device=base_weight.device, dtype=base_weight.dtype)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _check_raw(self, raw, ndim):
        require(isinstance(raw, torch.Tensor) and raw.ndim == ndim and
                raw.shape[-1] == self.input_dim and all(size > 0 for size in raw.shape),
                "Invalid raw vector shape")
        require(raw.device == self.base_weight.device and raw.dtype == self.base_weight.dtype,
                "Raw vector device/dtype must match the model")

    def prepare_queries(self, raw_queries):
        """Project [B,D] cached query means once, returning normalized base/head."""
        self._check_raw(raw_queries, 2)
        require(bool(torch.isfinite(raw_queries).all()), "Nonfinite query vectors")
        return (F.normalize(F.linear(raw_queries, self.base_weight), dim=-1),
                F.normalize(self.head(raw_queries), dim=-1))

    def prepare_candidates(self, raw_evidence, mask, raw_means=None):
        """Prepare [C,M,D] evidence; optional [C,D] means preserve cache pooling.

        Masked slots are replaced before any projection or averaging, so their
        values (including nonfinite padding) never affect scores or gradients.
        """
        self._check_raw(raw_evidence, 3)
        require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and
                mask.shape == raw_evidence.shape[:2] and mask.device == raw_evidence.device,
                "Evidence mask must be bool [candidate,evidence] on the vector device")
        require(bool(mask.any(dim=1).all()), "Every candidate requires real evidence")
        clean = raw_evidence.masked_fill(~mask.unsqueeze(-1), 0.0)
        require(bool(torch.isfinite(clean).all()), "Nonfinite real evidence vectors")
        if raw_means is None:
            raw_means = clean.sum(dim=1) / mask.sum(dim=1, keepdim=True).to(clean.dtype)
        else:
            self._check_raw(raw_means, 2)
            require(raw_means.shape[0] == raw_evidence.shape[0] and
                    bool(torch.isfinite(raw_means).all()), "Invalid supplied candidate means")
        base = F.normalize(F.linear(raw_means, self.base_weight), dim=-1)
        evidence = F.normalize(self.head(clean), dim=-1)
        return base, evidence, mask

    def score_prepared(self, query_tuple, candidate_tuple, return_diagnostics=False):
        """Score all prepared query/candidate pairs without per-chunk GPU sync.

        Tuple slicing along the candidate dimension is supported. Diagnostics
        return device tensors for the two sums and a Python integer pair count.
        """
        base_q, head_q = query_tuple
        base_c, head_e, mask = candidate_tuple
        require(base_q.ndim == 2 and head_q.ndim == 2 and base_c.ndim == 2 and head_e.ndim == 3 and
                base_q.shape == (head_q.shape[0], self.input_dim) and
                head_q.shape[1] == self.hidden_dim and
                base_c.shape == (head_e.shape[0], self.input_dim) and
                head_e.shape[2] == self.hidden_dim and mask.shape == head_e.shape[:2] and
                mask.dtype == torch.bool and head_q.shape[0] > 0 and head_e.shape[0] > 0,
                "Invalid prepared vector shapes")
        count = mask.sum(dim=-1)
        if self.mode == "attention":
            logits = torch.einsum("bh,cmh->bcm", head_q, head_e) / self.attention_temperature
            logits = logits.masked_fill(~mask.unsqueeze(0), -torch.inf)
            weights = logits.softmax(dim=-1)
        else:
            weights = (mask.to(head_e.dtype) / count.unsqueeze(-1)).unsqueeze(0).expand(
                head_q.shape[0], -1, -1)
        context = F.normalize(torch.einsum("bcm,cmh->bch", weights, head_e), dim=-1)
        query = head_q.unsqueeze(1).expand_as(context)
        features = torch.cat((query, context, query * context, (query - context).abs()), dim=-1)
        residual = self.residual_scale * torch.tanh(self.mlp(features).squeeze(-1))
        scores = base_q @ base_c.T + residual
        if not return_diagnostics:
            return scores
        entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(dim=-1)
        denominator = count.to(weights.dtype).log().clamp_min(1.0e-12)
        normalized_entropy = torch.where(count.unsqueeze(0) > 1,
                                         entropy / denominator.unsqueeze(0), torch.zeros_like(entropy))
        diagnostics = {
            "normalized_attention_entropy_sum": normalized_entropy.sum().detach(),
            "attention_max_weight_sum": weights.max(dim=-1).values.sum().detach(),
            "attention_pair_count": head_q.shape[0] * head_e.shape[0],
        }
        return scores, diagnostics

    def forward(self, raw_queries, raw_evidence, mask):
        return self.score_prepared(self.prepare_queries(raw_queries),
                                   self.prepare_candidates(raw_evidence, mask))
