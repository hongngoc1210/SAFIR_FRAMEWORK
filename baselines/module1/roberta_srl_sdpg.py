from __future__ import annotations

import ast
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, accuracy_score,
    precision_score, recall_score,
)
import pickle

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

ENCODER_NAME = "hfl/chinese-roberta-wwm-ext"
HIDDEN_SIZE  = 768
NUM_FACTORS  = 10
NUM_CLASSES  = 3
MAX_LEN      = 192
MAX_NEWS     = 8
BATCH_SIZE   = 8
EPOCHS       = 15
LR_ENCODER   = 2e-5
LR_HEAD      = 1e-3
DROPOUT      = 0.1
MLP_HIDDEN   = 1024
WARMUP_RATIO = 0.1
PATIENCE     = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LTP WRAPPER — SRL + SDPG
# ═══════════════════════════════════════════════════════════════════════════════

class LTPProcessor:
    def __init__(self, ltp_model: str = "LTP/small"):
        from ltp import LTP
        print(f"[LTP] Loading {ltp_model} ...")
        self.ltp = LTP(ltp_model)
        if torch.cuda.is_available():
            self.ltp.to("cuda")
        print("[LTP] Ready.")

    @staticmethod
    def _parse_srl_matrix(
        srl_matrix: List[List[str]],
    ) -> Dict[str, List[int]]:
        roles: Dict[str, List[int]] = {"V": [], "A0": [], "A1": []}

        for pred_idx, tags in enumerate(srl_matrix):
            has_pred = any(t in ("B-PRED", "PRED") for t in tags)
            if not has_pred:
                continue

            for word_idx, tag in enumerate(tags):
                if tag in ("B-PRED", "PRED"):
                    roles["V"].append(word_idx)
                elif tag in ("B-ARG0", "I-ARG0"):
                    roles["A0"].append(word_idx)
                elif tag in ("B-ARG1", "I-ARG1"):
                    roles["A1"].append(word_idx)

            if roles["V"]:
                break

        return roles

    @staticmethod
    def _filter_sdpg_edges(
        sdpg_edges : List[Tuple[int, int, str]],
        roles      : Dict[str, List[int]],
    ) -> Dict[str, List[Tuple[int, int]]]:
        edges: Dict[str, List[Tuple[int, int]]] = {
            "G_VA0": [], "G_VA1": [], "G_A0A1": []
        }

        v_set  = set(roles["V"])
        a0_set = set(roles["A0"])
        a1_set = set(roles["A1"])

        for h1, d1, _rel in sdpg_edges:
            h = h1 - 1
            d = d1 - 1
            if h < 0 or d < 0:
                continue

            pairs = [(h, d), (d, h)]
            for a, b in pairs:
                if a in v_set  and b in a0_set: edges["G_VA0" ].append((a, b))
                if a in v_set  and b in a1_set: edges["G_VA1" ].append((a, b))
                if a in a0_set and b in a1_set: edges["G_A0A1"].append((a, b))

        return edges

    def process(self, texts: List[str]) -> List[Dict]:
        output = self.ltp.pipeline(texts, tasks=["cws", "srl", "sdpg"])

        results = []
        for i in range(len(texts)):
            tokens     = output.cws[i]
            srl_matrix = output.srl[i]  if output.srl  else []
            sdpg_edges = output.sdpg[i] if output.sdpg else []

            roles = self._parse_srl_matrix(srl_matrix)
            edges = self._filter_sdpg_edges(sdpg_edges, roles)

            results.append({
                "tokens": tokens,
                "srl"   : roles,
                "sdpg"  : edges,
            })

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["stock_factors"] = df["stock_factors"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)
    return df


def group_by_stock_date(df: pd.DataFrame) -> List[dict]:
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (str(row["CODE"]), str(row["trade_date"]))
        groups[key].append(row)

    samples = []
    for (code, trade_date), rows in groups.items():
        rows = rows[:MAX_NEWS]
        samples.append({
            "texts"        : [r["text_a"] for r in rows],
            "stock_factors": rows[0]["stock_factors"],
            "label"        : int(rows[0]["label"]),
            "code"         : code,
            "trade_date"   : trade_date,
        })
    return samples


# FIX #2: Bỏ WeightedRandomSampler vì data cân bằng
# — dùng shuffle=True trong DataLoader thay thế


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class NewsFactorDataset(Dataset):
    def __init__(
        self,
        samples       : List[dict],
        tokenizer     : BertTokenizerFast,
        ltp_processor : LTPProcessor,
        max_length    : int = MAX_LEN,
        cache_path    : str = "srl_cache.pkl",
    ):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.ltp        = ltp_processor
        self.max_length = max_length

        if os.path.exists(cache_path):
            print(f"📦 Loading SRL cache from {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            self.ltp_cache = cache_data["ltp_cache"]
            print(f"✅ Loaded {len(self.ltp_cache)} cached samples")
        else:
            print("Pre-processing SRL + SDPG ...")
            all_texts = [t for s in samples for t in s["texts"]]
            all_ltp   = self._batch_ltp(all_texts)

            idx = 0
            self.ltp_cache: List[List[Dict]] = []
            for s in samples:
                n = len(s["texts"])
                self.ltp_cache.append(all_ltp[idx : idx + n])
                idx += n

            print(f"✅ Done. {len(all_texts)} texts processed.")
            with open(cache_path, "wb") as f:
                pickle.dump({"ltp_cache": self.ltp_cache}, f)
            print(f"💾 Cache saved → {cache_path}")

    def _batch_ltp(self, texts: List[str], chunk: int = 64) -> List[Dict]:
        results = []
        for i in range(0, len(texts), chunk):
            batch = texts[i : i + chunk]
            results.extend(self.ltp.process(batch))
        return results

    def _srl_mask(
        self,
        role_indices : List[int],
        word_ids     : List[Optional[int]],
    ) -> torch.Tensor:
        role_set = set(role_indices)
        mask = torch.zeros(self.max_length, dtype=torch.float)
        for tok_pos, word_id in enumerate(word_ids):
            if tok_pos >= self.max_length:
                break
            if word_id is not None and word_id in role_set:
                mask[tok_pos] = 1.0
        return mask

    @staticmethod
    def _sdpg_feat(edges: Dict[str, List[Tuple[int, int]]]) -> torch.Tensor:
        feat = []
        for key in ("G_VA0", "G_VA1", "G_A0A1"):
            e = edges.get(key, [])
            if e:
                feat.append(float(len(e)))
                feat.append(float(np.mean([h for h, _ in e])))
            else:
                feat.extend([0.0, 0.0])
        return torch.tensor(feat, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s        = self.samples[idx]
        ltp_list = self.ltp_cache[idx]

        all_input_ids = []
        all_attn_mask = []
        all_mask_V    = []
        all_mask_A0   = []
        all_mask_A1   = []
        all_sdpg      = []

        for text, ltp_out in zip(s["texts"], ltp_list):
            enc_fast = self.tokenizer(
                ltp_out["tokens"],
                is_split_into_words=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            word_ids = enc_fast.word_ids(batch_index=0)

            all_input_ids.append(enc_fast["input_ids"].squeeze(0))
            all_attn_mask.append(enc_fast["attention_mask"].squeeze(0))
            all_mask_V .append(self._srl_mask(ltp_out["srl"]["V"],  word_ids))
            all_mask_A0.append(self._srl_mask(ltp_out["srl"]["A0"], word_ids))
            all_mask_A1.append(self._srl_mask(ltp_out["srl"]["A1"], word_ids))
            all_sdpg   .append(self._sdpg_feat(ltp_out["sdpg"]))

        return {
            "input_ids"     : torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attn_mask),
            "mask_V"        : torch.stack(all_mask_V),
            "mask_A0"       : torch.stack(all_mask_A0),
            "mask_A1"       : torch.stack(all_mask_A1),
            "sdpg_feat"     : torch.stack(all_sdpg),
            "stock_factors" : torch.tensor(s["stock_factors"], dtype=torch.float32),
            "label"         : torch.tensor(s["label"],         dtype=torch.long),
            "code"          : s["code"],
            "trade_date"    : s["trade_date"],
        }


# ── collate ──────────────────────────────────────────────────────────────────

def collate_fn(batch):
    max_N = max(b["input_ids"].size(0) for b in batch)

    def pad_news(key, dtype=torch.long):
        tensors = []
        for b in batch:
            N = b[key].size(0)
            pad_shape = [max_N - N] + list(b[key].shape[1:])
            tensors.append(torch.cat([b[key], torch.zeros(pad_shape, dtype=dtype)], dim=0))
        return torch.stack(tensors)

    news_counts = torch.tensor([b["input_ids"].size(0) for b in batch], dtype=torch.long)

    return {
        "input_ids"     : pad_news("input_ids"),
        "attention_mask": pad_news("attention_mask"),
        "mask_V"        : pad_news("mask_V",      torch.float),
        "mask_A0"       : pad_news("mask_A0",     torch.float),
        "mask_A1"       : pad_news("mask_A1",     torch.float),
        "sdpg_feat"     : pad_news("sdpg_feat",   torch.float),
        "news_counts"   : news_counts,
        "stock_factors" : torch.stack([b["stock_factors"] for b in batch]),
        "label"         : torch.stack([b["label"]          for b in batch]),
        "code"          : [b["code"]       for b in batch],
        "trade_date"    : [b["trade_date"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL
# ═══════════════════════════════════════════════════════════════════════════════

SDPG_FEAT_DIM = 6


class NewsFactorizationModule(nn.Module):
    def __init__(
        self,
        encoder_name  : str   = ENCODER_NAME,
        hidden_size   : int   = HIDDEN_SIZE,
        num_factors   : int   = NUM_FACTORS,
        num_classes   : int   = NUM_CLASSES,
        mlp_hidden    : int   = MLP_HIDDEN,
        dropout       : float = DROPOUT,
        news_proj_dim : int   = 256,
        use_grad_ckpt : bool  = True,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_factors   = num_factors
        self.news_proj_dim = news_proj_dim

        print(f"[Model] Loading encoder: {encoder_name}")
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if use_grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
            print("[Model] Gradient checkpointing ENABLED.")

        self.role_attn = nn.Linear(hidden_size, 1, bias=False)

        self.sdpg_proj = nn.Sequential(
            nn.Linear(SDPG_FEAT_DIM, hidden_size),
            nn.Tanh(),
        )

        self.xn_proj = nn.Sequential(
            nn.Linear(5 * hidden_size, news_proj_dim),
            nn.LayerNorm(news_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.factor_norm = nn.LayerNorm(num_factors)

        self._xn_xf_dim = news_proj_dim + num_factors
        self.W_alpha = nn.Parameter(torch.zeros(self._xn_xf_dim))

        self.mlp = nn.Sequential(
            nn.Linear(self._xn_xf_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(mlp_hidden // 2, num_classes),
        )

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def _attn_pool(
        self,
        hidden : torch.Tensor,
        mask   : torch.Tensor,
    ) -> torch.Tensor:
        scores = self.role_attn(hidden).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        valid = mask.sum(dim=-1, keepdim=True) > 0
        scores = torch.where(
            valid.expand_as(scores),
            scores,
            torch.zeros_like(scores),
        )
        weights = F.softmax(scores, dim=-1)
        pooled  = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        pooled  = pooled * valid.float()
        return pooled

    def _encode_news(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
        mask_V         : torch.Tensor,
        mask_A0        : torch.Tensor,
        mask_A1        : torch.Tensor,
        sdpg_feat      : torch.Tensor,
        news_counts    : torch.Tensor,
    ) -> torch.Tensor:

        B, N_max, L = input_ids.shape

        flat_ids   = input_ids.view(B * N_max, L)
        flat_masks = attention_mask.view(B * N_max, L)

        hidden = self.encoder(
            input_ids=flat_ids,
            attention_mask=flat_masks,
        ).last_hidden_state

        e_cls = hidden[:, 0, :]

        flat_V  = mask_V .view(B * N_max, L)
        flat_A0 = mask_A0.view(B * N_max, L)
        flat_A1 = mask_A1.view(B * N_max, L)

        e_V  = self._attn_pool(hidden, flat_V)
        e_A0 = self._attn_pool(hidden, flat_A0)
        e_A1 = self._attn_pool(hidden, flat_A1)

        flat_sdpg = sdpg_feat.view(B * N_max, SDPG_FEAT_DIM)
        e_sdpg    = self.sdpg_proj(flat_sdpg)

        x_n_raw = torch.cat([e_V, e_A0, e_A1, e_sdpg, e_cls], dim=-1)
        x_n     = self.xn_proj(x_n_raw)
        x_n     = x_n.view(B, N_max, self.news_proj_dim)

        return x_n

    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
        mask_V         : torch.Tensor,
        mask_A0        : torch.Tensor,
        mask_A1        : torch.Tensor,
        sdpg_feat      : torch.Tensor,
        news_counts    : torch.Tensor,
        stock_factors  : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N_max, _ = input_ids.shape

        x_n = self._encode_news(
            input_ids, attention_mask,
            mask_V, mask_A0, mask_A1,
            sdpg_feat, news_counts,
        )

        x_f = self.factor_norm(stock_factors)
        x_f = x_f.unsqueeze(1).expand(B, N_max, self.num_factors)

        x_concat = torch.cat([x_n, x_f], dim=-1)
        W = torch.sigmoid(self.W_alpha)
        X = x_concat * W.unsqueeze(0).unsqueeze(0)

        count_mask = (
            torch.arange(N_max, device=X.device)
            .unsqueeze(0).lt(news_counts.unsqueeze(1))
        ).float().unsqueeze(-1)

        logits_per_news = self.mlp(X)
        logits = (logits_per_news * count_mask).sum(1) \
                 / count_mask.sum(1).clamp(min=1)

        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def get_alpha_stats(self):
        W = torch.sigmoid(self.W_alpha).detach()
        w_news   = W[:self.news_proj_dim].mean().item()
        w_factor = W[self.news_proj_dim:].mean().item()
        return w_news, w_factor


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def alpha_diversity_loss(
    model      : NewsFactorizationModule,
    lambda_div : float = 0.05,
) -> torch.Tensor:
    W = torch.sigmoid(model.W_alpha)
    w_news   = W[:model.news_proj_dim]
    w_factor = W[model.news_proj_dim:]
    balance  = (w_news.mean() - w_factor.mean()) ** 2
    eps = 1e-6
    W_c = W.clamp(eps, 1 - eps)
    entropy = -(W_c * W_c.log() + (1 - W_c) * (1 - W_c).log()).mean()
    return lambda_div * balance - 0.01 * entropy


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def _batch_to_device(batch, device):
    keys = [
        "input_ids", "attention_mask",
        "mask_V", "mask_A0", "mask_A1",
        "sdpg_feat", "news_counts",
        "stock_factors", "label",
    ]
    return {k: batch[k].to(device) for k in keys}


# FIX #1: Accumulate preds/labels toàn epoch thay vì chỉ batch cuối
# FIX #2 (KL): Giảm KL weight 0.5 → 0.1
def train_epoch(model, loader, optimizer, device, lambda_div=0.05):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []                          # FIX #1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        b = _batch_to_device(batch, device)
        optimizer.zero_grad()

        logits1, _ = model(
            b["input_ids"], b["attention_mask"],
            b["mask_V"], b["mask_A0"], b["mask_A1"],
            b["sdpg_feat"], b["news_counts"], b["stock_factors"],
        )
        logits2, _ = model(
            b["input_ids"], b["attention_mask"],
            b["mask_V"], b["mask_A0"], b["mask_A1"],
            b["sdpg_feat"], b["news_counts"], b["stock_factors"],
        )

        task_loss = (criterion(logits1, b["label"]) +
                     criterion(logits2, b["label"])) / 2

        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        kl = (F.kl_div(p1, p2.exp(), reduction="batchmean") +
              F.kl_div(p2, p1.exp(), reduction="batchmean")) / 2

        div  = alpha_diversity_loss(model, lambda_div)
        loss = task_loss + 0.1 * kl + div                  # FIX KL: 0.5 → 0.1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += task_loss.item() * len(b["label"])
        correct    += (logits1.argmax(-1) == b["label"]).sum().item()
        total      += len(b["label"])

        # FIX #1: accumulate toàn epoch
        all_preds.extend(logits1.argmax(-1).cpu().tolist())
        all_labels.extend(b["label"].cpu().tolist())

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    # FIX #1: tính F1 trên toàn epoch
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, correct / total, epoch_f1


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        b = _batch_to_device(batch, device)
        logits, _ = model(
            b["input_ids"], b["attention_mask"],
            b["mask_V"], b["mask_A0"], b["mask_A1"],
            b["sdpg_feat"], b["news_counts"], b["stock_factors"],
        )
        loss  = criterion(logits, b["label"])
        preds = logits.argmax(-1)

        total_loss += loss.item() * len(b["label"])
        correct    += (preds == b["label"]).sum().item()
        total      += len(b["label"])
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(b["label"].cpu().tolist())

    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec       = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec        = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    report     = classification_report(
        all_labels, all_preds,
        target_names=["Negative", "Neutral", "Positive"],
        digits=4, zero_division=0,
    )
    conf_mat   = confusion_matrix(all_labels, all_preds)

    return total_loss / total, correct / total, prec, rec, macro_f1, report, conf_mat


# ── Export features ───────────────────────────────────────────────────────────

@torch.no_grad()
def export_features_to_csv(model, loader, device, output_path="features.csv"):
    """
    FIX #3: Encode hanya 1x per batch — reuse hidden dari forward
    Export: code, trade_date, news_idx, label, pred_label,
            pred_prob_neg/neu/pos, cls_emb (768-dim)
    """
    model.eval()
    all_records = []

    print(f"Extracting features → {output_path}")
    for batch in tqdm(loader, desc="Exporting"):
        b = _batch_to_device(batch, device)
        B, N_max, L = b["input_ids"].shape

        # Encode 1 lần để lấy CLS embedding
        flat_ids   = b["input_ids"].view(B * N_max, L)
        flat_masks = b["attention_mask"].view(B * N_max, L)
        hidden     = model.encoder(
            input_ids=flat_ids,
            attention_mask=flat_masks,
        ).last_hidden_state                              # (B*N, L, H)
        cls_emb    = hidden[:, 0, :].view(B, N_max, -1) # (B, N, H)

        # Forward lần 2 để lấy logits/probs
        # (encoder chạy 2 lần — tradeoff để giữ code sạch;
        #  nếu cần tối ưu VRAM, refactor forward() nhận hidden từ ngoài)
        logits, probs = model(
            b["input_ids"], b["attention_mask"],
            b["mask_V"], b["mask_A0"], b["mask_A1"],
            b["sdpg_feat"], b["news_counts"], b["stock_factors"],
        )

        probs_cpu   = probs.cpu().tolist()
        pred_labels = logits.argmax(-1).cpu().tolist()
        labels      = batch["label"].tolist()
        news_counts = b["news_counts"].cpu().tolist()

        for i in range(B):
            real_n = int(news_counts[i])
            p = probs_cpu[i]
            for j in range(real_n):
                all_records.append({
                    "CODE"         : batch["code"][i],
                    "trade_date"   : batch["trade_date"][i],
                    "news_idx"     : j,
                    "label"        : labels[i],
                    "pred_label"   : pred_labels[i],
                    "pred_prob_neg": round(p[0], 6),
                    "pred_prob_neu": round(p[1], 6),
                    "pred_prob_pos": round(p[2], 6),
                    "cls_emb"      : cls_emb[i, j].cpu().tolist(),
                })

    pd.DataFrame(all_records).to_csv(output_path, index=False)
    print(f"✅ {len(all_records)} rows saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_f1):
    torch.save({
        "model"    : model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch"    : epoch,
        "best_val" : best_val_f1,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer : optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler : scheduler.load_state_dict(ckpt["scheduler"])
    print(f"✅ Loaded epoch={ckpt['epoch']}  best_val_f1={ckpt['best_val']:.4f}")
    return ckpt["epoch"], ckpt["best_val"]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    TRAIN_PATH = "/kaggle/input/datasets/phmhngtrang/module1-finreport/train_module1.csv"
    VAL_PATH   = "/kaggle/input/datasets/phmhngtrang/module1-finreport/val_module1.csv"
    TEST_PATH  = "/kaggle/input/datasets/phmhngtrang/module1-finreport/test_module1.csv"

    CKPT_PATH  = "checkpoint_finreport_m1.pt"
    BEST_PATH  = "best_finreport_m1.pt"
    RESUME     = True
    LAMBDA_DIV = 0.05
    WARMUP_EP  = 2

    set_seed(42)
    print(f"Device: {DEVICE}")

    # ── Load data ─────────────────────────────────────────────────────────────
    samples_train = group_by_stock_date(load_data(TRAIN_PATH))
    samples_val   = group_by_stock_date(load_data(VAL_PATH))
    samples_test  = group_by_stock_date(load_data(TEST_PATH))
    print(f"Train={len(samples_train)}  Val={len(samples_val)}  Test={len(samples_test)}")

    # ── LTP + Tokenizer ───────────────────────────────────────────────────────
    ltp       = LTPProcessor("LTP/small")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

    train_ds = NewsFactorDataset(samples_train, tokenizer, ltp, cache_path="train_srl_cache.pkl")
    val_ds   = NewsFactorDataset(samples_val,   tokenizer, ltp, cache_path="val_srl_cache.pkl")
    test_ds  = NewsFactorDataset(samples_test,  tokenizer, ltp, cache_path="test_srl_cache.pkl")

    # FIX #2: shuffle=True thay vì WeightedRandomSampler vì data cân bằng
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    # ── Detect num_factors ────────────────────────────────────────────────────
    actual_num_factors = len(samples_train[0]["stock_factors"])
    print(f"Detected num_factors = {actual_num_factors}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NewsFactorizationModule(
        encoder_name  = ENCODER_NAME,
        hidden_size   = HIDDEN_SIZE,
        num_factors   = actual_num_factors,
        num_classes   = NUM_CLASSES,
        mlp_hidden    = MLP_HIDDEN,
        dropout       = DROPOUT,
        news_proj_dim = 256,
        use_grad_ckpt = True,
    ).to(DEVICE)

    # ── Differential LR ───────────────────────────────────────────────────────
    enc_params  = list(model.encoder.parameters())
    enc_ids     = {id(p) for p in enc_params}
    alpha_param = [model.W_alpha]
    alpha_ids   = {id(p) for p in alpha_param}
    head_params = [p for p in model.parameters()
                   if id(p) not in enc_ids and id(p) not in alpha_ids]

    optimizer = torch.optim.AdamW([
        {"params": enc_params,  "lr": LR_ENCODER, "weight_decay": 1e-2},
        {"params": head_params, "lr": LR_HEAD,    "weight_decay": 1e-2},
        {"params": alpha_param, "lr": LR_HEAD,    "weight_decay": 0.0},
    ])

    def lr_lambda(epoch):
        if epoch < WARMUP_EP:
            return (epoch + 1) / max(WARMUP_EP, 1)
        progress = (epoch - WARMUP_EP) / max(EPOCHS - WARMUP_EP, 1)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))  # floor 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────────
    # FIX #4: best_val theo F1 (cao hơn = tốt hơn) thay vì loss
    best_val   = 0.0        # FIX #4: 0.0 thay vì float("inf")
    start_ep   = 1
    patience_c = 0

    model.freeze_encoder()
    print(f"🔒 Encoder frozen for {WARMUP_EP} warmup epochs.")

    if RESUME and os.path.exists(CKPT_PATH):
        start_ep, best_val = load_checkpoint(CKPT_PATH, model, optimizer, scheduler, DEVICE)
        start_ep += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_ep, EPOCHS + 1):
        if epoch == WARMUP_EP + 1:
            model.unfreeze_encoder()
            print(f"\n🔓 Encoder unfrozen at epoch {epoch}")

        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss, tr_acc, tr_f1              = train_epoch(model, train_loader, optimizer, DEVICE, LAMBDA_DIV)
        vl_loss, vl_acc, _, _, vl_f1, vl_report, vl_conf = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        w_n, w_f = model.get_alpha_stats()
        print(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  f1={tr_f1:.4f}")
        print(f"  val    loss={vl_loss:.4f}  acc={vl_acc:.4f}  f1={vl_f1:.4f}")
        print(f"  W_α    news={w_n:.4f}  factor={w_f:.4f}  ratio={w_n/(w_f+1e-8):.3f}")

        # FIX #4: monitor val F1 thay vì val loss
        if vl_f1 > best_val:
            best_val   = vl_f1
            patience_c = 0
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  ✅ Best saved  (f1={vl_f1:.4f}  acc={vl_acc:.4f})")
            print(f"  Val Classification Report:\n{vl_report}")
            print(f"  Val Confusion Matrix:\n{vl_conf}")
        else:
            patience_c += 1
            print(f"  ⏳ No improvement ({patience_c}/{PATIENCE})")

        save_checkpoint(CKPT_PATH, model, optimizer, scheduler, epoch, best_val)

        if patience_c >= PATIENCE:
            print(f"\n⚠ Early stopping at epoch {epoch}.")
            break

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
    ts_loss, ts_acc, ts_prec, ts_rec, ts_f1, ts_report, ts_conf = evaluate(model, test_loader, DEVICE)
    print(f"\n📊 Test  loss={ts_loss:.4f}  acc={ts_acc:.4f}  "
          f"precision={ts_prec:.4f}  recall={ts_rec:.4f}  macro_f1={ts_f1:.4f}")
    print(f"  Classification Report:\n{ts_report}")
    print(f"  Confusion Matrix:\n{ts_conf}")

    # ── Export features ───────────────────────────────────────────────────────
    print("\n📤 Exporting features...")
    export_features_to_csv(model, train_loader, DEVICE, "roberta_srl_sdpg_features_train.csv")
    export_features_to_csv(model, val_loader,   DEVICE, "roberta_srl_sdpg_features_val.csv")
    export_features_to_csv(model, test_loader,  DEVICE, "roberta_srl_sdpg_features_test.csv")


if __name__ == "__main__":
    main()