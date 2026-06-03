from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config as cfg


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class NewsPriceFusion(nn.Module):
    """Bidirectional news-price cross-attention fusion.

    Price is a real temporal sequence. News_factor is currently a daily stock-aware
    vector, so it is broadcast across the lookback window. If future preprocessing
    provides a news sequence with shape (B, T, news_dim), this module accepts it too.
    """

    def __init__(self):
        super().__init__()
        self.price_proj = nn.Sequential(
            nn.Linear(cfg.price_feat_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.news_proj = nn.Sequential(
            nn.Linear(cfg.news_factor_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.price_to_news = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.news_to_price = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, price_seq: torch.Tensor, news_factor: torch.Tensor) -> torch.Tensor:
        B, T, _ = price_seq.shape
        price_emb = self.price_proj(price_seq)  # (B, T, D)

        if news_factor.dim() == 2:
            news_seq = news_factor.unsqueeze(1).expand(B, T, -1)
        elif news_factor.dim() == 3:
            if news_factor.size(1) != T:
                raise ValueError("news sequence length must match price_seq length")
            news_seq = news_factor
        else:
            raise ValueError("news_factor must have shape (B, D) or (B, T, D)")
        news_emb = self.news_proj(news_seq)

        if cfg.use_bidirectional_fusion:
            p2n, _ = self.price_to_news(query=price_emb, key=news_emb, value=news_emb)
            n2p, _ = self.news_to_price(query=news_emb, key=price_emb, value=price_emb)
            fused = torch.cat([price_emb, news_emb, p2n, n2p], dim=-1)
        else:
            zeros = torch.zeros_like(price_emb)
            fused = torch.cat([price_emb, news_emb, zeros, zeros], dim=-1)
        return self.fuse(fused)


class CausalConvBlock(nn.Module):
    """Temporal causal convolution. Output at t cannot use information after t."""

    def __init__(self, d_model: int = cfg.d_model, kernel_size: int = cfg.causal_kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        y = self.conv(x_t).transpose(1, 2)
        return self.norm(residual + self.dropout(F.gelu(y)))


class MarketGuidedGating(nn.Module):
    """MASTER-inspired feature gate from market status.

    Uses softmax competition rather than sigmoid, closer to MASTER's feature
    selection idea. Multiplication by d_model keeps the expected scale around 1.
    """

    def __init__(self, mkt_dim: int = cfg.mkt_dim, d_model: int = cfg.d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(mkt_dim, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, mkt_vector: torch.Tensor) -> torch.Tensor:
        logits = self.proj(mkt_vector)
        temp = self.temperature.clamp_min(0.05)
        return cfg.d_model * torch.softmax(logits / temp, dim=-1)


class MarketGatedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = cfg.d_model,
        n_heads: int = cfg.n_heads,
        d_ff: int = cfg.d_ff,
        dropout: float = cfg.dropout,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, gate: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = residual + attn_out * gate.unsqueeze(1)
        residual = x
        x = residual + self.ff(self.norm2(x))
        return x


class RelationGCN(nn.Module):
    """Optional relation-graph fusion over stocks.

    Disabled by default. Only call set_stock_adjacency(adj) when code_ids index a
    valid stock relation matrix. This avoids using random batch order as a graph.
    """

    def __init__(self, d_model: int = cfg.d_model, dropout: float = cfg.graph_dropout):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("stock_adj", torch.empty(0), persistent=False)

    def set_adjacency(self, adj: torch.Tensor) -> None:
        if adj.dim() != 2 or adj.size(0) != adj.size(1):
            raise ValueError("adjacency must be a square matrix")
        self.stock_adj = adj.float()

    def forward(self, reps: torch.Tensor, code_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if code_ids is None or self.stock_adj.numel() == 0:
            return reps
        max_id = int(code_ids.max().item()) if code_ids.numel() else 0
        if max_id >= self.stock_adj.size(0):
            return reps
        A = self.stock_adj.to(reps.device)[code_ids][:, code_ids]
        A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        h = A @ reps
        h = self.dropout(F.gelu(self.proj(h)))
        return self.norm(reps + h)


class RiskEstimatorHead(nn.Module):
    def __init__(self, in_dim: int = cfg.d_model, hidden: int = cfg.egarch_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        log_vol = out[:, 0]
        var_est = F.softplus(out[:, 1])
        return log_vol, var_est


class DMAQModule(nn.Module):
    """Module II - v3 Stock-aware News-Price Fusion Forecasting Core."""

    def __init__(self, n_codes: int = 5000):
        super().__init__()
        self.fusion = NewsPriceFusion()
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.lookback, dropout=cfg.dropout)
        self.causal_conv = CausalConvBlock(cfg.d_model, cfg.causal_kernel_size)
        self.market_gate = MarketGuidedGating()
        self.blocks = nn.ModuleList([MarketGatedTransformerBlock() for _ in range(cfg.n_layers)])
        self.temporal_query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.temporal_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.relation_gcn = RelationGCN()
        self.risk_head = RiskEstimatorHead()
        self.return_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.n_classes),
        )

    def set_stock_adjacency(self, adj: torch.Tensor) -> None:
        self.relation_gcn.set_adjacency(adj)

    def forward(
        self,
        price_seq: torch.Tensor,
        mkt_vector: torch.Tensor,
        news_factor: torch.Tensor,
        code_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.fusion(price_seq, news_factor)
        x = self.pos_enc(x)
        x = self.causal_conv(x)

        gate = self.market_gate(mkt_vector)
        for block in self.blocks:
            x = block(x, gate)

        # Temporal aggregation: the last representation and a learned query both
        # summarize the sequence. This is more stable than only taking x[:, -1, :].
        q = self.temporal_query.expand(x.size(0), -1, -1)
        temporal_rep, _ = self.temporal_attn(query=q, key=x, value=x)
        rep = 0.5 * x[:, -1, :] + 0.5 * temporal_rep.squeeze(1)
        rep = self.relation_gcn(rep, code_ids)

        log_vol, var_est = self.risk_head(rep)
        logits = self.return_head(rep)

        return {
            "logits": logits,
            "quant_factors": rep,
            "log_vol": log_vol,
            "var_est": var_est,
            "gate_weights": gate,
        }
