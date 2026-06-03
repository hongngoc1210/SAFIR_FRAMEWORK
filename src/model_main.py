from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import Config as cfg
from .model_dmaq import DMAQModule
from .model_isf import ISFModule
from .model_sefn import SEFNModule, compute_reward


class FinReportNextGen(nn.Module):
    """SAFIR: Stock-aware news pooling -> news-price fusion -> report layer."""

    def __init__(self, n_codes: int = 5000, enable_sefn: Optional[bool] = None):
        super().__init__()
        self.isf = ISFModule(n_codes=n_codes)
        self.dmaq = DMAQModule(n_codes=n_codes)
        self.enable_sefn = cfg.enable_sefn if enable_sefn is None else enable_sefn
        self.sefn = SEFNModule() if self.enable_sefn else None

    def set_stock_adjacency(self, adj: torch.Tensor) -> None:
        self.dmaq.set_stock_adjacency(adj)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        code_ids: torch.Tensor,
        price_seq: torch.Tensor,
        mkt_vector: torch.Tensor,
        news_item_mask: Optional[torch.Tensor] = None,
        true_labels: Optional[torch.Tensor] = None,
        generate_text: bool = False,
    ) -> Dict[str, torch.Tensor]:
        news_factor = self.isf(
            input_ids=input_ids,
            attention_mask=attention_mask,
            code_ids=code_ids,
            news_item_mask=news_item_mask,
        )
        dmaq_out = self.dmaq(price_seq, mkt_vector, news_factor, code_ids=code_ids)

        logits = dmaq_out["logits"]
        pred_labels = logits.argmax(dim=-1)
        confidence = logits.softmax(dim=-1).max(dim=-1).values

        explanations = None
        rewards = None
        if true_labels is not None:
            rewards = compute_reward(
                pred_labels=pred_labels,
                true_labels=true_labels,
                quant_factors=dmaq_out["quant_factors"],
            )

        if generate_text and self.sefn is not None:
            explanations = self.sefn.generate_explanation(
                quant_factors=dmaq_out["quant_factors"],
                pred_label=pred_labels,
                true_label=true_labels,
                confidence=confidence,
                log_vol=dmaq_out["log_vol"],
                var_est=dmaq_out["var_est"],
            )

        return {
            "logits": logits,
            "pred_labels": pred_labels,
            "confidence": confidence,
            "news_factor": news_factor,
            "quant_factors": dmaq_out["quant_factors"],
            "log_vol": dmaq_out["log_vol"],
            "var_est": dmaq_out["var_est"],
            "gate_weights": dmaq_out["gate_weights"],
            "explanations": explanations,
            "rewards": rewards,
        }
