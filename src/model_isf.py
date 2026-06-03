from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertModel

from .config import Config as cfg


class StockAwareNewsPooling(nn.Module):
    """Stock-aware pooling over a daily collection of news embeddings.

    Supports three modes inspired by the stockvsnews paper:
    - cap    : stock embedding is Query, news embeddings are Keys/Values.
    - sap    : stock embedding is appended to the news sequence, then queried.
    - pa_sap : news embeddings are augmented with stock and positional embeddings.

    Input article_emb is already a document-level embedding per news item, usually the
    BERT [CLS] representation. This is intentionally lighter than feeding every article
    token into cross-attention, and is better suited for small academic datasets.
    """

    def __init__(
        self,
        bert_dim: int = cfg.bert_dim,
        code_dim: int = cfg.code_emb_dim,
        n_heads: int = cfg.sap_heads,
        out_dim: int = cfg.news_factor_dim,
        max_news: int = cfg.max_news_per_day,
        mode: str = cfg.news_pooling,
    ):
        super().__init__()
        if mode not in {"cap", "sap", "pa_sap"}:
            raise ValueError("news pooling mode must be one of: cap, sap, pa_sap")
        self.mode = mode
        self.stock_proj = nn.Linear(code_dim, bert_dim)
        self.attn = nn.MultiheadAttention(bert_dim, n_heads, batch_first=True)
        self.norm_news = nn.LayerNorm(bert_dim)
        self.norm_out = nn.LayerNorm(bert_dim)
        self.position_emb = nn.Embedding(max_news, bert_dim)
        self.global_query = nn.Parameter(torch.randn(1, 1, bert_dim) * 0.02)
        self.out_proj = nn.Sequential(
            nn.Linear(bert_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        article_emb: torch.Tensor,
        code_emb: torch.Tensor,
        news_item_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool daily news articles into one stock-conditioned news factor.

        Args:
            article_emb: (B, N, bert_dim)
            code_emb: (B, code_dim)
            news_item_mask: (B, N), 1 for valid articles, 0 for padded slots.
        """
        B, N, _ = article_emb.shape
        article_emb = self.norm_news(article_emb)
        stock_token = self.stock_proj(code_emb).unsqueeze(1)  # (B, 1, D)

        safe_mask = news_item_mask.bool().clone()
        empty_rows = ~safe_mask.any(dim=-1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True

        if self.mode == "cap":
            query = stock_token
            key_value = article_emb
            key_padding_mask = ~safe_mask
        elif self.mode == "sap":
            query = stock_token
            key_value = torch.cat([stock_token, article_emb], dim=1)
            stock_valid = torch.ones(B, 1, dtype=torch.bool, device=article_emb.device)
            key_padding_mask = ~torch.cat([stock_valid, safe_mask], dim=1)
        else:  # pa_sap
            pos_ids = torch.arange(N, device=article_emb.device).unsqueeze(0).expand(B, N)
            pos = self.position_emb(pos_ids)
            augmented_news = article_emb + stock_token.expand(-1, N, -1) + pos
            query = self.global_query.expand(B, -1, -1)
            key_value = augmented_news
            key_padding_mask = ~safe_mask

        attn_out, attn_weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        pooled = self.norm_out(attn_out.squeeze(1))
        pooled = self.out_proj(pooled)
        pooled = pooled * news_item_mask.any(dim=-1, keepdim=True).float()
        return pooled, attn_weights


class ISFModule(nn.Module):
    """Module I - Stock-aware daily news filtering.

    Accepted input shapes:
      - input_ids / attention_mask: (B, L) for legacy single-text input.
      - input_ids / attention_mask: (B, N, L) for v3 daily-news collection input.
    """

    def __init__(self, n_codes: int = 5000):
        super().__init__()
        self.bert = BertModel.from_pretrained(cfg.bert_model)
        self._freeze_bert_layers(freeze_up_to=10)

        if cfg.grad_checkpointing:
            self.bert.gradient_checkpointing_enable()

        self.code_embedding = nn.Embedding(n_codes, cfg.code_emb_dim, padding_idx=0)
        self.news_pool = StockAwareNewsPooling()

    def _freeze_bert_layers(self, freeze_up_to: int = 10) -> None:
        if not hasattr(self.bert, "encoder"):
            return
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_up_to:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        code_ids: torch.Tensor,
        news_item_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        # Backward compatibility: (B, L) -> (B, 1, L)
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
            if news_item_mask is None:
                news_item_mask = attention_mask.any(dim=-1).long()
        elif input_ids.dim() == 3:
            if news_item_mask is None:
                news_item_mask = attention_mask.any(dim=-1).long()
        else:
            raise ValueError("input_ids must have shape (B, L) or (B, N, L)")

        B, N, L = input_ids.shape
        flat_ids = input_ids.reshape(B * N, L)
        flat_mask = attention_mask.reshape(B * N, L)

        # BERT can produce NaN when every token in a row is masked. Safely unmask
        # the first token for empty rows, then discard these article embeddings via
        # news_item_mask during pooling.
        safe_flat_mask = flat_mask.clone()
        empty_articles = ~safe_flat_mask.any(dim=-1)
        if empty_articles.any():
            safe_flat_mask[empty_articles, 0] = 1

        bert_out = self.bert(input_ids=flat_ids, attention_mask=safe_flat_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :].reshape(B, N, cfg.bert_dim)
        article_valid = news_item_mask.to(cls_emb.device).bool()
        cls_emb = cls_emb * article_valid.unsqueeze(-1).float()

        code_emb = self.code_embedding(code_ids)
        news_factor, attn_weights = self.news_pool(
            article_emb=cls_emb,
            code_emb=code_emb,
            news_item_mask=article_valid.long(),
        )

        if return_attn:
            return news_factor, attn_weights
        return news_factor
