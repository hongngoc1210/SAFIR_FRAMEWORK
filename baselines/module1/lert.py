from __future__ import annotations

import ast
import warnings
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import os

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME  = "hfl/chinese-lert-large"
HIDDEN_SIZE = 1024
MAX_NEWS    = 8
DROPOUT     = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & GROUPING
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['stock_factors'] = df['stock_factors'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date.astype(str)
    return df


def group_by_stock_date(df: pd.DataFrame) -> List[dict]:
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (str(row["CODE"]), str(row["trade_date"]))
        groups[key].append(row)

    samples = []
    for (code, trade_date), rows in groups.items():
        rows = sorted(rows, key=lambda x: x["DATE"], reverse=True)
        rows = rows[:MAX_NEWS]

        samples.append({
            "texts"        : [r["text_a"] for r in rows],
            "stock_factors": rows[0]["stock_factors"],
            "label"        : int(rows[0]["label"]),
            "code"         : code,
            "trade_date"   : trade_date,
        })
    return samples


def make_sampler(samples: List[dict], seed: int = 42) -> WeightedRandomSampler:
    labels        = [s['label'] for s in samples]
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class NewsFactorDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        all_input_ids, all_attn_masks = [], []

        for text in s['texts']:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            all_input_ids.append(enc['input_ids'].squeeze(0))
            all_attn_masks.append(enc['attention_mask'].squeeze(0))

        return {
            'input_ids':      torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attn_masks),
            'stock_factors':  torch.tensor(s['stock_factors'], dtype=torch.float32),
            'label':          torch.tensor(s['label'], dtype=torch.long),
            'code':           s['code'],
            'trade_date':     s['trade_date'],
        }


def collate_fn(batch):
    max_N = max(b['input_ids'].size(0) for b in batch)
    L     = batch[0]['input_ids'].size(1)
    padded_ids, padded_masks, news_counts = [], [], []

    for b in batch:
        N   = b['input_ids'].size(0)
        pad = max_N - N
        news_counts.append(N)
        padded_ids.append(
            torch.cat([b['input_ids'],   torch.zeros(pad, L, dtype=torch.long)], dim=0)
        )
        padded_masks.append(
            torch.cat([b['attention_mask'], torch.zeros(pad, L, dtype=torch.long)], dim=0)
        )

    return {
        'input_ids':      torch.stack(padded_ids),
        'attention_mask': torch.stack(padded_masks),
        'news_counts':    torch.tensor(news_counts, dtype=torch.long),
        'stock_factors':  torch.stack([b['stock_factors'] for b in batch]),
        'label':          torch.stack([b['label']          for b in batch]),
        'code':           [b['code']       for b in batch],
        'trade_date':     [b['trade_date'] for b in batch],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class NewsAttentionAggregator(nn.Module):
    """
    FIX: Thay thế masked mean bằng learned attention aggregation.

    Vấn đề gốc:
    - Masked mean aggregate logits (sau MLP) → mỗi news được weight bằng nhau
    - Không capture được: news nào quan trọng hơn cho quyết định

    Giải pháp:
    - Attention score từ projected feature → soft weight cho từng news
    - Aggregate ở FEATURE LEVEL trước khi qua MLP classifier
    """
    def __init__(self, feat_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.Tanh(),
            nn.Linear(feat_dim // 4, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, news_counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X:           (B, N_max, D)
            news_counts: (B,)
        Returns:
            agg:         (B, D)  — attended aggregate
        """
        B, N_max, D = X.shape

        # Attention scores
        scores = self.attn_proj(X).squeeze(-1)              # (B, N_max)

        # Mask padding positions với -inf
        mask = torch.arange(N_max, device=X.device).unsqueeze(0) \
                    .ge(news_counts.unsqueeze(1))            # (B, N_max) — True ở pad
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)            # (B, N_max)
        attn_weights = self.dropout(attn_weights)

        agg = (attn_weights.unsqueeze(-1) * X).sum(dim=1)   # (B, D)
        return agg


class NewsFactorizationModule(nn.Module):
    """
    Module 1 — News Factorization với LERT-Large (cải tiến).

    Các thay đổi so với bản gốc:
    1. W_alpha khởi tạo với inductive bias (news > factor ban đầu)
    2. Aggregate ở feature level (không phải logit level)
    3. NewsAttentionAggregator thay masked mean
    4. Factor projection riêng → cross-attention với news features
    5. Residual connection trong MLP head
    """

    def __init__(
        self,
        model_name  : str   = MODEL_NAME,
        hidden_size : int   = HIDDEN_SIZE,
        num_factors : int   = 24,
        num_classes : int   = 3,
        mlp_hidden  : int   = 512,
        dropout     : float = 0.3,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_factors = num_factors
        self._total_feat_dim = hidden_size + num_factors

        # ── LERT encoder ──────────────────────────────────────────────────────
        print(f"[Model] Loading: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            print("[Model] Gradient checkpointing ENABLED.")

        # ── Factor projection ─────────────────────────────────────────────────
        # FIX: normalize + project factors lên không gian cao hơn
        self.factor_norm = nn.LayerNorm(num_factors)
        self.factor_proj = nn.Sequential(
            nn.Linear(num_factors, num_factors * 2),
            nn.GELU(),
            nn.Linear(num_factors * 2, num_factors),
        )

        # ── W_alpha (Eq. 4) ───────────────────────────────────────────────────
        # FIX: khởi tạo với bias → news features được weight cao hơn ban đầu
        # sigmoid(1.0) ≈ 0.73 cho news dims, sigmoid(-0.5) ≈ 0.38 cho factor dims
        # W_init = torch.cat([
        #     torch.ones(hidden_size) * 1.0,    # news dims: ~0.73
        #     torch.ones(num_factors) * -0.5,   # factor dims: ~0.38
        # ])

        W_init = torch.cat([
            torch.ones(hidden_size) * 0.8,
            torch.ones(num_factors) * -0.2,
        ])
        self.W_alpha = nn.Parameter(W_init)

        # ── Attention Aggregator ───────────────────────────────────────────────
        # FIX: learned attention thay vì masked mean
        self.aggregator = NewsAttentionAggregator(
            feat_dim=self._total_feat_dim,
            dropout=dropout * 0.5,
        )

        # ── MLP classifier (với residual) ─────────────────────────────────────
        # FIX: aggregate ở feature level rồi mới classify (không phải logit level)
        self.mlp_in  = nn.Sequential(
            nn.Linear(self._total_feat_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mlp_mid = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.mlp_out = nn.Linear(mlp_hidden // 2, num_classes)

        # Residual projection (mlp_hidden → mlp_hidden//2)
        self.res_proj = nn.Linear(mlp_hidden, mlp_hidden // 2, bias=False)

    # ── Freeze / Unfreeze ─────────────────────────────────────────────────────

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def get_alpha_stats(self):
        W = torch.sigmoid(self.W_alpha).detach()
        w_news   = W[:self.hidden_size].mean().item()
        w_factor = W[self.hidden_size:].mean().item()
        return w_news, w_factor

    # ── Core encode ───────────────────────────────────────────────────────────

    def _encode(self, input_ids, attention_mask, news_counts, stock_factors):
        B, N_max, L = input_ids.shape

        # 1. Encode từng news → [CLS] embedding
        flat_ids   = input_ids.view(B * N_max, L)
        flat_masks = attention_mask.view(B * N_max, L)
        cls_emb = self.encoder(
            input_ids=flat_ids, attention_mask=flat_masks
        ).last_hidden_state[:, 0, :]                                   # (B*N_max, H)
        cls_emb = cls_emb.view(B, N_max, self.hidden_size)            # (B, N_max, H)

        # 2. Factor embedding + project
        factors_normed  = self.factor_norm(stock_factors)             # (B, F)
        factors_proj    = self.factor_proj(factors_normed)            # (B, F)
        X_f = factors_proj.unsqueeze(1).expand(B, N_max, self.num_factors)

        # 3. Eq.(4): W_alpha gating
        X_concat = torch.cat([cls_emb, X_f], dim=-1)                  # (B, N_max, H+F)
        W = torch.sigmoid(self.W_alpha)
        X = X_concat * W.unsqueeze(0).unsqueeze(0)                    # (B, N_max, H+F)

        # 4. FIX: Attended aggregate ở FEATURE LEVEL
        agg = self.aggregator(X, news_counts)                         # (B, H+F)

        # 5. FIX: MLP với residual connection
        h    = self.mlp_in(agg)                                       # (B, mlp_hidden)
        mid  = self.mlp_mid(h) + self.res_proj(h)                    # (B, mlp_hidden//2)
        logits = self.mlp_out(mid)                                    # (B, C)
        probs  = F.softmax(logits, dim=-1)

        return dict(logits=logits, probs=probs, cls_emb=cls_emb)

    def forward(self, input_ids, attention_mask, news_counts, stock_factors):
        out = self._encode(input_ids, attention_mask, news_counts, stock_factors)
        return out['logits'], out['probs']


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DIVERSITY REGULARIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def alpha_diversity_loss(model: NewsFactorizationModule, lambda_div: float = 0.05) -> torch.Tensor:
    W = torch.sigmoid(model.W_alpha)
    w_news   = W[:model.hidden_size]
    w_factor = W[model.hidden_size:]

    balance_loss = (w_news.mean() - w_factor.mean()) ** 2

    eps  = 1e-6
    W_c  = W.clamp(eps, 1 - eps)
    entropy_loss = -(W_c * W_c.log() + (1 - W_c) * (1 - W_c).log()).mean()

    return lambda_div * balance_loss - 0.01 * entropy_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, lambda_div=0.05):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    # FIX: Focal Loss thay CrossEntropyLoss + label_smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        input_ids     = batch['input_ids'].to(device)
        attn_mask     = batch['attention_mask'].to(device)
        news_counts   = batch['news_counts'].to(device)
        stock_factors = batch['stock_factors'].to(device)
        labels        = batch['label'].to(device)

        optimizer.zero_grad()

        # R-Drop: 2 forward passes
        logits1, _ = model(input_ids, attn_mask, news_counts, stock_factors)
        logits2, _ = model(input_ids, attn_mask, news_counts, stock_factors)

        task_loss = (criterion(logits1, labels) + criterion(logits2, labels)) / 2

        # FIX: symmetric KL với weight nhỏ hơn (0.1 thay vì 0.5)
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        kl_loss = (
            F.kl_div(p1, p2.detach().exp(), reduction='batchmean') +
            F.kl_div(p2, p1.detach().exp(), reduction='batchmean')
        ) / 2

        div_loss = alpha_diversity_loss(model, lambda_div)

        # FIX: giảm KL weight từ 0.5 → 0.1
        loss = task_loss + 0.1 * kl_loss + div_loss

        loss.backward()

        # FIX: tăng max_norm từ 0.3 → 1.0 (0.3 quá nhỏ, encoder không học được)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += task_loss.item() * len(labels)
        correct    += (logits1.argmax(-1) == labels).sum().item()
        total      += len(labels)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for batch in pbar:
        input_ids     = batch['input_ids'].to(device)
        attn_mask     = batch['attention_mask'].to(device)
        news_counts   = batch['news_counts'].to(device)
        stock_factors = batch['stock_factors'].to(device)
        labels        = batch['label'].to(device)

        logits, _ = model(input_ids, attn_mask, news_counts, stock_factors)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    classification_rep = classification_report(all_labels, all_preds, zero_division=0)
    return total_loss / total, correct / total, prec, rec, macro_f1, classification_rep


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  EXPORT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def export_features_to_csv(model, loader, device, output_path="features.csv"):
    model.eval()
    all_records = []

    print(f"Extracting features → {output_path}")
    for batch in tqdm(loader, desc="Exporting"):
        input_ids     = batch['input_ids'].to(device)
        attn_mask     = batch['attention_mask'].to(device)
        news_counts   = batch['news_counts'].to(device)
        stock_factors = batch['stock_factors'].to(device)
        labels        = batch['label']

        B   = input_ids.size(0)
        enc = model._encode(input_ids, attn_mask, news_counts, stock_factors)

        probs_cpu   = enc['probs'].cpu().tolist()
        pred_labels = enc['logits'].argmax(-1).cpu().tolist()

        for i in range(B):
            real_n = news_counts[i].item()
            p      = probs_cpu[i]
            for j in range(real_n):
                all_records.append({
                    'CODE':          batch['code'][i],
                    'trade_date':    batch['trade_date'][i],
                    'news_idx':      j,
                    'label':         labels[i].item(),
                    'pred_label':    pred_labels[i],
                    'pred_prob_neg': round(p[0], 6),
                    'pred_prob_neu': round(p[1], 6),
                    'pred_prob_pos': round(p[2], 6),
                    'cls_emb':       enc['cls_emb'][i, j].cpu().tolist(),
                })

    pd.DataFrame(all_records).to_csv(output_path, index=False)
    print(f"✅ {len(all_records)} rows saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CHECKPOINT UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch':                epoch,
        'best_val':             best_val,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch    = ckpt.get('epoch', 0)
    best_val = ckpt.get('best_val', 0.0)
    print(f"✅ Loaded checkpoint — epoch {epoch} | best_val={best_val:.4f}")
    return epoch, best_val


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    TRAIN_PATH = '/kaggle/input/datasets/lngivy/module1-finreport/train_module1.csv'
    VAL_PATH   = '/kaggle/input/datasets/lngivy/module1-finreport/val_module1.csv'
    TEST_PATH  = '/kaggle/input/datasets/lngivy/module1-finreport/test_module1.csv'

    CHECKPOINT_PATH = 'checkpoint_lert.pt'
    BEST_MODEL_PATH = 'best_lert.pt'
    RESUME = True

    MAX_LEN        = 192
    BATCH_SIZE     = 8
    EPOCHS         = 15
    LAMBDA_DIV     = 0.05
    WARMUP_EPOCHS  = 0
    PATIENCE       = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}  |  Model: {MODEL_NAME}")
    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    # ── Data ──────────────────────────────────────────────────────────────────
    samples_train = group_by_stock_date(load_data(TRAIN_PATH))
    samples_val   = group_by_stock_date(load_data(VAL_PATH))
    samples_test  = group_by_stock_date(load_data(TEST_PATH))

    num_factors = len(samples_train[0]['stock_factors'])
    print(f"Factors: {num_factors} | Train: {len(samples_train)} | Val: {len(samples_val)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = NewsFactorDataset(samples_train, tokenizer, MAX_LEN)
    val_ds   = NewsFactorDataset(samples_val,   tokenizer, MAX_LEN)
    test_ds  = NewsFactorDataset(samples_test,  tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=make_sampler(samples_train),
        num_workers=2, collate_fn=collate_fn, pin_memory=(DEVICE == 'cuda'),
        generator=g,
    )
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, collate_fn=collate_fn, pin_memory=(DEVICE == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, collate_fn=collate_fn, pin_memory=(DEVICE == 'cuda'))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NewsFactorizationModule(
        model_name  = MODEL_NAME,
        hidden_size = HIDDEN_SIZE,
        num_factors = num_factors,
        num_classes = 3,
        mlp_hidden  = 512,
        dropout     = DROPOUT,
        use_gradient_checkpointing = True,
    ).to(DEVICE)

    def log_alpha():
        w_n, w_f = model.get_alpha_stats()
        print(f"  W_alpha: news={w_n:.4f}  factor={w_f:.4f}  ratio={w_n/(w_f+1e-8):.3f}")

    # ── Optimizer — differential LR ───────────────────────────────────────────
    enc_params  = list(model.encoder.parameters())
    enc_ids     = {id(p) for p in enc_params}
    alpha_param = [model.W_alpha]
    alpha_ids   = {id(p) for p in alpha_param}
    head_params = [p for p in model.parameters()
                   if id(p) not in enc_ids and id(p) not in alpha_ids]

    # FIX: tăng encoder LR từ 2e-6 → 5e-6 (2e-6 quá nhỏ, encoder hầu như không cập nhật)
    optimizer = torch.optim.AdamW([
        {'params': enc_params,   'lr': 5e-6, 'weight_decay': 1e-2},
        {'params': head_params,  'lr': 5e-4, 'weight_decay': 1e-2},
        {'params': alpha_param,  'lr': 5e-4, 'weight_decay': 0.0},
    ])

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────────
    if WARMUP_EPOCHS > 0:
        print(f"\n🔒 Freezing encoder for {WARMUP_EPOCHS} warmup epochs")
        model.freeze_encoder()
    else:
        print("\n Encoder unfrozen from the start.")

    best_val         = 0.0
    patience_counter = 0
    start_epoch      = 1

    if RESUME and os.path.exists(CHECKPOINT_PATH):
        start_epoch, best_val = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, DEVICE
        )
        start_epoch += 1

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            print(f"\n🔓 Unfreezing encoder at epoch {epoch}")
            model.unfreeze_encoder()

        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc                = train_epoch(model, train_loader, optimizer, DEVICE, LAMBDA_DIV)
        val_loss,   val_acc, _, _, val_f1, _ = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        print(f"  train → loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  val   → loss={val_loss:.4f}  acc={val_acc:.4f}  macro_f1={val_f1:.4f}")
        log_alpha()

        if val_f1 > best_val:
            best_val         = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  → ✅ Best saved (loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{PATIENCE})")

        save_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, epoch, best_val)

        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping at epoch {epoch}.")
            break

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_report = evaluate(model, test_loader, DEVICE)

    print(f"\nTest | loss={test_loss:.4f}  acc={test_acc:.4f}  "
          f"precision={test_prec:.4f}  recall={test_rec:.4f}  macro_f1={test_f1:.4f}")
    print(f"Test Report:\n{test_report}")
    log_alpha()


if __name__ == '__main__':
    main()