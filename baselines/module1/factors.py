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
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

NUM_FACTORS = 24
NUM_CLASSES = 3
DROPOUT     = 0.3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  UTILS
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  GROUP — mỗi (CODE, trade_date) là 1 sample, lấy factors của dòng đầu tiên
# ═══════════════════════════════════════════════════════════════════════════════

def group_by_stock_date(df: pd.DataFrame) -> List[dict]:
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (str(row["CODE"]), str(row["trade_date"]))
        groups[key].append(row)

    samples = []
    for (code, trade_date), rows in groups.items():
        rows = sorted(rows, key=lambda x: x["DATE"], reverse=True)
        samples.append({
            "stock_factors": rows[0]["stock_factors"],   # list[float] len=24
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
# 3.  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class FactorDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'stock_factors': torch.tensor(s['stock_factors'], dtype=torch.float32),
            'label'        : torch.tensor(s['label'],         dtype=torch.long),
            'code'         : s['code'],
            'trade_date'   : s['trade_date'],
        }


def collate_fn(batch):
    return {
        'stock_factors': torch.stack([b['stock_factors'] for b in batch]),
        'label'        : torch.stack([b['label']          for b in batch]),
        'code'         : [b['code']       for b in batch],
        'trade_date'   : [b['trade_date'] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL  — MLP trên X_f  (mirror Eq.5 trong paper, chỉ dùng factors)
# ═══════════════════════════════════════════════════════════════════════════════

class FactorOnlyModule(nn.Module):
    """
    Eq.(5) của paper nhưng X chỉ gồm X_f (stock factors).

    Pipeline:
        X_f  →  LayerNorm
             →  MLP (2 layers + softmax)
             →  logits (3 classes)
    """

    def __init__(
        self,
        num_factors: int   = NUM_FACTORS,
        num_classes: int   = NUM_CLASSES,
        mlp_hidden : int   = 128,
        dropout    : float = DROPOUT,
    ):
        super().__init__()

        self.factor_norm = nn.LayerNorm(num_factors)

        # 2-layer MLP + Softmax (Eq. 5)
        self.mlp = nn.Sequential(
            nn.Linear(num_factors, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(mlp_hidden // 2, num_classes),
        )

    def forward(self, stock_factors: torch.Tensor):
        """
        Args:
            stock_factors: (B, num_factors)
        Returns:
            logits: (B, num_classes)
            probs : (B, num_classes)
        """
        x      = self.factor_norm(stock_factors)   # (B, F)
        logits = self.mlp(x)                       # (B, C)
        probs  = F.softmax(logits, dim=-1)
        return logits, probs


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        stock_factors = batch['stock_factors'].to(device)
        labels        = batch['label'].to(device)

        optimizer.zero_grad()

        # R-Drop: 2 forward passes với dropout khác nhau
        logits1, _ = model(stock_factors)
        logits2, _ = model(stock_factors)

        task_loss = (criterion(logits1, labels) + criterion(logits2, labels)) / 2

        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        kl_loss = (
            F.kl_div(p1, p2.exp(), reduction='batchmean') +
            F.kl_div(p2, p1.exp(), reduction='batchmean')
        ) / 2

        loss = task_loss + 0.5 * kl_loss
        loss.backward()
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
        stock_factors = batch['stock_factors'].to(device)
        labels        = batch['label'].to(device)

        logits, _ = model(stock_factors)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    prec     = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec      = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report   = classification_report(all_labels, all_preds, zero_division=0)
    return total_loss / total, correct / total, prec, rec, macro_f1, report


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CHECKPOINT UTILS
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
# 7.  EXPORT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def export_features_to_csv(model, loader, device, output_path="factor_features.csv"):
    model.eval()
    all_records = []

    for batch in tqdm(loader, desc=f"Exporting → {output_path}"):
        stock_factors = batch['stock_factors'].to(device)
        labels        = batch['label']

        logits, probs = model(stock_factors)
        pred_labels   = logits.argmax(-1).cpu().tolist()
        probs_cpu     = probs.cpu().tolist()

        for i in range(len(pred_labels)):
            p = probs_cpu[i]
            all_records.append({
                'CODE':          batch['code'][i],
                'trade_date':    batch['trade_date'][i],
                'label':         labels[i].item(),
                'pred_label':    pred_labels[i],
                'pred_prob_neg': round(p[0], 6),
                'pred_prob_neu': round(p[1], 6),
                'pred_prob_pos': round(p[2], 6),
            })

    pd.DataFrame(all_records).to_csv(output_path, index=False)
    print(f"✅ {len(all_records)} rows saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    TRAIN_PATH = '/kaggle/input/datasets/lngivy/module1-finreport/train_module1.csv'
    VAL_PATH   = '/kaggle/input/datasets/lngivy/module1-finreport/val_module1.csv'
    TEST_PATH  = '/kaggle/input/datasets/lngivy/module1-finreport/test_module1.csv'

    CHECKPOINT_PATH = 'checkpoint_factor_only.pt'
    BEST_MODEL_PATH = 'best_factor_only.pt'
    RESUME = True

    BATCH_SIZE  = 64     # lớn hơn vì không có encoder nặng
    EPOCHS      = 30
    PATIENCE    = 5
    MLP_HIDDEN  = 128

    print(f"Device: {DEVICE}  |  Mode: Factor-Only (no text encoder)")
    set_seed(42)

    # ── Data ─────────────────────────────────────────────────────────────────
    samples_train = group_by_stock_date(load_data(TRAIN_PATH))
    samples_val   = group_by_stock_date(load_data(VAL_PATH))
    samples_test  = group_by_stock_date(load_data(TEST_PATH))

    num_factors = len(samples_train[0]['stock_factors'])
    print(f"Factors: {num_factors} | Train: {len(samples_train)} | Val: {len(samples_val)} | Test: {len(samples_test)}")

    train_ds = FactorDataset(samples_train)
    val_ds   = FactorDataset(samples_val)
    test_ds  = FactorDataset(samples_test)

    g = torch.Generator(); g.manual_seed(42)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=make_sampler(samples_train),
        num_workers=2, collate_fn=collate_fn, pin_memory=(DEVICE == 'cuda'), generator=g,
    )
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, collate_fn=collate_fn, pin_memory=(DEVICE == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, collate_fn=collate_fn, pin_memory=(DEVICE == 'cuda'))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FactorOnlyModule(
        num_factors = num_factors,
        num_classes = NUM_CLASSES,
        mlp_hidden  = MLP_HIDDEN,
        dropout     = DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def lr_lambda(epoch):
        progress = epoch / max(EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────────
    best_val         = 0.0
    patience_counter = 0
    start_epoch      = 1

    if RESUME and os.path.exists(CHECKPOINT_PATH):
        start_epoch, best_val = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, DEVICE
        )
        start_epoch += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc                = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_acc, _, _, val_f1, _   = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        print(f"  train → loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  val   → loss={val_loss:.4f}  acc={val_acc:.4f}  macro_f1={val_f1:.4f}")

        if val_f1 > best_val:
            best_val         = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  → ✅ Best saved (f1={val_f1:.4f})")
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

    print(f"\n{'='*60}")
    print(f"Test | loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(f"     | precision={test_prec:.4f}  recall={test_rec:.4f}  macro_f1={test_f1:.4f}")
    print(f"Test Report:\n{test_report}")


if __name__ == '__main__':
    main()