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

MODEL_NAME  = "hfl/chinese-roberta-wwm-ext"   # RoBERTa 
HIDDEN_SIZE = 768
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
# 3.  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class NewsFactorizationModule(nn.Module):
    def __init__(
        self,
        model_name:  str   = MODEL_NAME,
        hidden_size: int   = HIDDEN_SIZE,
        num_classes: int   = 3,
        mlp_hidden:  int   = 384,
        dropout:     float = 0.3,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        print(f"[Model] Loading: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            print("[Model] Gradient checkpointing ENABLED.")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
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

    def _encode(self, input_ids, attention_mask, news_counts):
        B, N_max, L = input_ids.shape

        flat_ids   = input_ids.view(B * N_max, L)
        flat_masks = attention_mask.view(B * N_max, L)
        cls_emb = self.encoder(
            input_ids=flat_ids, attention_mask=flat_masks
        ).last_hidden_state[:, 0, :]                        # (B*N_max, H)
        cls_emb = cls_emb.view(B, N_max, self.hidden_size)  # (B, N_max, H)

        # Masked mean aggregate
        count_mask = (
            torch.arange(N_max, device=cls_emb.device)
            .unsqueeze(0).lt(news_counts.unsqueeze(1))
        )                                                    # (B, N_max)
        mask_float = count_mask.unsqueeze(-1).float()
        logits_per_news = self.mlp(cls_emb)                 # (B, N_max, C)
        logits = (logits_per_news * mask_float).sum(dim=1) \
                 / mask_float.sum(dim=1).clamp(min=1)        # (B, C)
        probs  = F.softmax(logits, dim=-1)

        return dict(logits=logits, probs=probs, cls_emb=cls_emb)

    def forward(self, input_ids, attention_mask, news_counts):
        out = self._encode(input_ids, attention_mask, news_counts)
        return out['logits'], out['probs']


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        input_ids   = batch['input_ids'].to(device)
        attn_mask   = batch['attention_mask'].to(device)
        news_counts = batch['news_counts'].to(device)
        labels      = batch['label'].to(device)

        optimizer.zero_grad()

        logits1, _ = model(input_ids, attn_mask, news_counts)
        logits2, _ = model(input_ids, attn_mask, news_counts)

        task_loss = (criterion(logits1, labels) + criterion(logits2, labels)) / 2

        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        kl_loss = (F.kl_div(p1, p2.exp(), reduction='batchmean') +
                   F.kl_div(p2, p1.exp(), reduction='batchmean')) / 2

        loss = task_loss + 0.5 * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
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
        input_ids   = batch['input_ids'].to(device)
        attn_mask   = batch['attention_mask'].to(device)
        news_counts = batch['news_counts'].to(device)
        labels      = batch['label'].to(device)

        logits, _ = model(input_ids, attn_mask, news_counts)
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
# 6.  EXPORT FEATURES
# ══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def export_features_to_csv(model, loader, device, output_path="features.csv"):
    model.eval()
    all_records = []

    print(f"Extracting features → {output_path}")
    for batch in tqdm(loader, desc="Exporting"):
        input_ids     = batch['input_ids'].to(device)
        attn_mask     = batch['attention_mask'].to(device)
        news_counts   = batch['news_counts'].to(device)
        labels        = batch['label']

        B   = input_ids.size(0)
        enc = model._encode(input_ids, attn_mask, news_counts)

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
# 7.  CHECKPOINT UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch':                epoch,
        'best_val':        best_val,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch         = ckpt.get('epoch', 0)
    # best_val = ckpt.get('best_val', float('inf'))
    best_val = ckpt.get('best_val', 0.0)
    print(f"✅ Loaded checkpoint — epoch {epoch} | best_val={best_val:.4f}")
    return epoch, best_val


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    TRAIN_PATH = '/kaggle/input/datasets/lngivy/module1-finreport/train_module1.csv'
    VAL_PATH   = '/kaggle/input/datasets/lngivy/module1-finreport/val_module1.csv'
    TEST_PATH  = '/kaggle/input/datasets/lngivy/module1-finreport/test_module1.csv'

    CHECKPOINT_PATH = 'checkpoint_roberta_news.pt'
    BEST_MODEL_PATH = 'best_roberta_news.pt'
    RESUME = True

    MAX_LEN            = 192
    BATCH_SIZE         = 8
    EPOCHS             = 15
    WARMUP_EPOCHS      = 0
    PATIENCE           = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}  |  Model: {MODEL_NAME}")
    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    # ── Data ─────────────────────────────────────────────────────────────────
    samples_train = group_by_stock_date(load_data(TRAIN_PATH))
    samples_val   = group_by_stock_date(load_data(VAL_PATH))
    samples_test  = group_by_stock_date(load_data(TEST_PATH))

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
        num_classes = 3,
        mlp_hidden  = 384,
        dropout     = DROPOUT,
        use_gradient_checkpointing = True,
    ).to(DEVICE)

    # ── Optimizer — differential LR ──────────────────────────────────────────
    enc_params  = list(model.encoder.parameters())
    enc_ids     = {id(p) for p in enc_params}

    head_params = [p for p in model.parameters()
                   if id(p) not in enc_ids]

    optimizer = torch.optim.AdamW([
        {'params': enc_params,   'lr': 5e-6, 'weight_decay': 1e-2},
        {'params': head_params,  'lr': 5e-4, 'weight_decay': 1e-2},
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

    # best_val    = float('inf')
    best_val    = 0.0
    patience_counter = 0
    start_epoch      = 1

    if RESUME and os.path.exists(CHECKPOINT_PATH):
        start_epoch, best_val = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, DEVICE
        )
        start_epoch += 1

    for epoch in range(start_epoch, EPOCHS + 1):
        if WARMUP_EPOCHS > 0 and epoch == WARMUP_EPOCHS + 1:
            print(f"\n🔓 Unfreezing encoder at epoch {epoch}")
            model.unfreeze_encoder()

        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc                = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss,   val_acc, _, _, val_f1, _ = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        print(f"  train → loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  val   → loss={val_loss:.4f}  acc={val_acc:.4f}  macro_f1={val_f1:.4f}")

        if val_f1 > best_val:
            best_val    = val_f1
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

    print(f"\nTest | loss={test_loss:.4f}  acc={test_acc:.4f}  precision={test_prec:.4f}  recall={test_rec:.4f}  macro_f1={test_f1:.4f}")
    print(f"Test Report:\n{test_report}")

    # ── Export features ───────────────────────────────────────────────────────
    # print("\n📤 Exporting features...")
    # export_features_to_csv(model, train_loader, DEVICE, "lert_features_train.csv")
    # export_features_to_csv(model, val_loader,   DEVICE, "lert_features_val.csv")
    # export_features_to_csv(model, test_loader,  DEVICE, "lert_features_test.csv")


if __name__ == '__main__':
    main()