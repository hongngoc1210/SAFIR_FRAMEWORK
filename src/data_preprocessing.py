"""
FinReport-NextGen v3 | Data Preprocessing Pipeline
==================================================
Key changes vs v2:
  1. Binary labels are the default because the current dataset uses {0, 1}.
  2. CODE -> code_id mapping is preserved and saved for inference.
  3. text_a can be a daily collection of multiple news items; the tokenizer returns
     (max_news_per_day, max_text_len) tensors and a news_item_mask.
  4. Risk targets are computed from the available price window for supervised risk loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from .config import Config as cfg

warnings.filterwarnings("ignore")

COLS_TO_DROP = ["Unnamed: 0", "READ", "DESCRIPTION"]
PRICE_FEATURES = ["open", "close"]


@dataclass
class CodeVocabulary:
    code_to_id: Dict[str, int]

    @classmethod
    def build(cls, *dfs: pd.DataFrame) -> "CodeVocabulary":
        codes = set()
        for df in dfs:
            if "CODE" in df.columns:
                codes.update(str(c) for c in df["CODE"].dropna().unique())
        code_to_id = {code: idx + 1 for idx, code in enumerate(sorted(codes))}
        return cls(code_to_id=code_to_id)

    def encode(self, code) -> int:
        return self.code_to_id.get(str(code), 0)

    @property
    def n_codes(self) -> int:
        return len(self.code_to_id) + 1


class RawDataCleaner:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> "RawDataCleaner":
        # The original FinReport data is tab-separated. If a comma-separated file is
        # passed, pandas will usually load one column; retry with comma in that case.
        df = pd.read_csv(self.filepath, low_memory=False, on_bad_lines="skip", sep="\t")
        if len(df.columns) == 1:
            df = pd.read_csv(self.filepath, low_memory=False, on_bad_lines="skip")
        self.df = df
        print(f"[Cleaner] Loaded {len(self.df):,} rows × {len(self.df.columns)} cols")
        return self

    def drop_legacy_cols(self) -> "RawDataCleaner":
        existing_drop = [c for c in COLS_TO_DROP if c in self.df.columns]
        self.df.drop(columns=existing_drop, inplace=True)
        print(f"[Cleaner] Dropped {len(existing_drop)} technical columns: {existing_drop}")
        return self

    def normalize_date(self) -> "RawDataCleaner":
        if "DATE" in self.df.columns:
            self.df["DATE"] = pd.to_datetime(self.df["DATE"])
            self.df["trade_date"] = self.df["DATE"].dt.date
        elif "trade_date" in self.df.columns:
            self.df["trade_date"] = pd.to_datetime(self.df["trade_date"]).dt.date
        else:
            raise ValueError("Input data must contain DATE or trade_date")
        return self

    def fill_missing_text(self) -> "RawDataCleaner":
        for col in ["text_a", "TITLE"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")
        if "text_a" not in self.df.columns:
            if "TITLE" in self.df.columns:
                self.df["text_a"] = self.df["TITLE"].fillna("")
            else:
                self.df["text_a"] = ""
        return self

    def get_df(self) -> pd.DataFrame:
        assert self.df is not None, "Call .load() first"
        return self.df.copy()


class MarketStatusBuilder:
    def __init__(self, df: pd.DataFrame, price_col_open: str = "open1", price_col_close: str = "close1"):
        self.df = df
        self.open_col = price_col_open
        self.close_col = price_col_close

    def compute(self) -> pd.DataFrame:
        df = self.df.copy()
        if self.open_col in df.columns and self.close_col in df.columns:
            open_ = pd.to_numeric(df[self.open_col], errors="coerce").replace(0, np.nan)
            close = pd.to_numeric(df[self.close_col], errors="coerce")
            df["_daily_ret"] = ((close - open_) / open_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            df["_daily_ret"] = 0.0

        mkt = df.groupby("trade_date")["_daily_ret"].agg(mkt_mean="mean", mkt_std="std").reset_index()
        mkt["mkt_std"] = mkt["mkt_std"].fillna(0.0)
        df = df.merge(mkt, on="trade_date", how="left")
        df.drop(columns=["_daily_ret"], inplace=True)
        print(f"[MarketStatus] Market status computed for {len(mkt)} trading days.")
        return df


class LookbackWindowBuilder:
    def __init__(self, df: pd.DataFrame, code_vocab: CodeVocabulary, lookback: int = cfg.lookback):
        self.df = df.sort_values(["CODE", "trade_date"]).reset_index(drop=True)
        self.lookback = lookback
        self.code_vocab = code_vocab

    def _extract_price_series_from_wide(self, row: pd.Series) -> np.ndarray:
        records = []
        for i in range(1, 6):
            o = row.get(f"open{i}", np.nan)
            c = row.get(f"close{i}", np.nan)
            records.append([
                float(o) if pd.notna(o) else 0.0,
                float(c) if pd.notna(c) else 0.0,
            ])
        return np.array(records, dtype=np.float32)

    def _pad_or_truncate(self, price_seq: np.ndarray) -> np.ndarray:
        pad_len = self.lookback - len(price_seq)
        if pad_len > 0:
            pad = np.zeros((pad_len, price_seq.shape[1]), dtype=np.float32)
            return np.concatenate([pad, price_seq], axis=0)
        return price_seq[-self.lookback :]

    def _encode_label(self, raw_label) -> int:
        try:
            y = float(raw_label)
        except Exception:
            y = 0.0

        if cfg.label_mode == "binary":
            return 1 if y > 0 else 0

        if y < 0:
            return 0
        if y > 0:
            return 2
        return 1

    def _compute_risk_targets(self, price_seq_raw: np.ndarray) -> Tuple[float, float]:
        open_ = price_seq_raw[:, 0]
        close = price_seq_raw[:, 1]
        valid = open_ > 1e-8
        if valid.sum() < 2:
            return float(np.log(1e-6)), 0.0
        ret = (close[valid] - open_[valid]) / (open_[valid] + 1e-8)
        mu = float(np.mean(ret))
        vol = float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0
        log_vol = float(np.log(vol + 1e-6))
        var_target = float(max(0.0, -(mu - cfg.var_confidence_z * vol)))
        return log_vol, var_target

    def build_sequences(self) -> List[Dict]:
        samples: List[Dict] = []
        for code, grp in self.df.groupby("CODE"):
            grp = grp.sort_values("trade_date").reset_index(drop=True)
            for _, row in grp.iterrows():
                raw_price_seq = self._extract_price_series_from_wide(row)
                log_vol_target, var_target = self._compute_risk_targets(raw_price_seq)
                price_seq = self._pad_or_truncate(raw_price_seq)
                label = self._encode_label(row.get("label", 0))

                samples.append(
                    {
                        "CODE": str(code),
                        "code_id": self.code_vocab.encode(code),
                        "trade_date": row["trade_date"],
                        "text_a": row.get("text_a", ""),
                        "price_seq": price_seq,
                        "mkt_mean": float(row.get("mkt_mean", 0.0)),
                        "mkt_std": float(row.get("mkt_std", 0.0)),
                        "label": int(label),
                        "log_vol_target": log_vol_target,
                        "var_target": var_target,
                    }
                )
        print(f"[Lookback] Built {len(samples):,} samples with lookback={self.lookback}")
        return samples


class NewsTokenizer:
    def __init__(self, model_name: str = cfg.bert_model, max_length: int = cfg.max_text_len, max_news: int = cfg.max_news_per_day):
        print(f"[Tokenizer] Loading tokenizer: {model_name}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        except Exception:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.max_news = max_news
        # Split common multi-news delimiters without breaking normal punctuation too much.
        self._splitter = re.compile(r"\s*(?:\n+|\|\|\||<SEP>|\[SEP\]|;\s*\d+\.|\t)\s*", flags=re.IGNORECASE)

    def split_news(self, text: str) -> List[str]:
        if not text or str(text).strip() == "":
            return []
        raw = str(text).strip()
        parts = [p.strip() for p in self._splitter.split(raw) if p and p.strip()]
        # If no delimiter is present, keep the full text as one article.
        if not parts:
            parts = [raw]
        return parts[: self.max_news]

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Backward-compatible single-text encoder."""
        if not text or str(text).strip() == "":
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "token_type_ids": torch.zeros(self.max_length, dtype=torch.long),
            }
        enc = self.tokenizer(
            str(text),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def encode_many(self, text: str) -> Dict[str, torch.Tensor]:
        articles = self.split_news(text)
        input_ids = torch.zeros((self.max_news, self.max_length), dtype=torch.long)
        attention_mask = torch.zeros((self.max_news, self.max_length), dtype=torch.long)
        news_item_mask = torch.zeros(self.max_news, dtype=torch.long)

        for i, article in enumerate(articles):
            enc = self.encode(article)
            input_ids[i] = enc["input_ids"]
            attention_mask[i] = enc["attention_mask"]
            news_item_mask[i] = 1 if enc["attention_mask"].any() else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "news_item_mask": news_item_mask,
        }


class FinReportDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        tokenizer: NewsTokenizer,
        code_vocab: CodeVocabulary,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.code_vocab = code_vocab

        all_prices = np.stack([s["price_seq"] for s in samples])
        N, T, F = all_prices.shape
        flat = all_prices.reshape(-1, F)

        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.scaler.fit(flat)
        else:
            self.scaler = scaler

        self.scaled_prices = self.scaler.transform(flat).reshape(N, T, F)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        enc = self.tokenizer.encode_many(s["text_a"])
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "news_item_mask": enc["news_item_mask"],
            "code_id": torch.tensor(s["code_id"], dtype=torch.long),
            "price_seq": torch.tensor(self.scaled_prices[idx], dtype=torch.float32),
            "mkt_vector": torch.tensor([s["mkt_mean"], s["mkt_std"]], dtype=torch.float32),
            "label": torch.tensor(s["label"], dtype=torch.long),
            "log_vol_target": torch.tensor(s["log_vol_target"], dtype=torch.float32),
            "var_target": torch.tensor(s["var_target"], dtype=torch.float32),
            "code": s["CODE"],
            "date": str(s["trade_date"]),
        }


def _load_clean_split(path: Path) -> pd.DataFrame:
    return (
        RawDataCleaner(str(path))
        .load()
        .drop_legacy_cols()
        .normalize_date()
        .fill_missing_text()
        .get_df()
    )


def _prepare_samples_from_df(df: pd.DataFrame, code_vocab: CodeVocabulary, lookback: int) -> List[Dict]:
    df = MarketStatusBuilder(df).compute()
    return LookbackWindowBuilder(df, code_vocab=code_vocab, lookback=lookback).build_sequences()


def _print_label_distribution(samples: List[Dict], name: str) -> None:
    counts = {i: 0 for i in range(cfg.n_classes)}
    for s in samples:
        counts[int(s["label"])] = counts.get(int(s["label"]), 0) + 1
    named = {cfg.label_names[k] if k < len(cfg.label_names) else str(k): v for k, v in counts.items()}
    print(f"[Labels] {name}: {named}")
    if any(v == 0 for v in counts.values()):
        print(f"[Labels Warning] {name} has zero-support class(es). Check cfg.label_mode and label construction.")


def run_preprocessing(
    raw_dir: str,
    bert_model: str = cfg.bert_model,
    lookback: int = cfg.lookback,
    max_text_len: int = cfg.max_text_len,
    batch_size: int = cfg.batch_size,
    return_metadata: bool = False,
):
    raw_path = Path(raw_dir).expanduser().resolve()
    train_file = raw_path / "train.csv"
    val_file = raw_path / "val.csv"
    test_file = raw_path / "test.csv"

    missing_files = [str(p) for p in [train_file, val_file, test_file] if not p.exists()]
    if missing_files:
        raise FileNotFoundError("Missing required split files: " + ", ".join(missing_files))

    train_df = _load_clean_split(train_file)
    val_df = _load_clean_split(val_file)
    test_df = _load_clean_split(test_file)

    code_vocab = CodeVocabulary.build(train_df, val_df, test_df)
    print(f"[CodeVocab] n_codes={code_vocab.n_codes:,} including UNK=0")

    train_s = _prepare_samples_from_df(train_df, code_vocab, lookback=lookback)
    val_s = _prepare_samples_from_df(val_df, code_vocab, lookback=lookback)
    test_s = _prepare_samples_from_df(test_df, code_vocab, lookback=lookback)

    _print_label_distribution(train_s, "Train")
    _print_label_distribution(val_s, "Val")
    _print_label_distribution(test_s, "Test")

    tokenizer = NewsTokenizer(model_name=bert_model, max_length=max_text_len, max_news=cfg.max_news_per_day)
    print(f"[Split] Train={len(train_s):,} | Val={len(val_s):,} | Test={len(test_s):,}")

    train_ds = FinReportDataset(train_s, tokenizer, code_vocab=code_vocab, fit_scaler=True)
    val_ds = FinReportDataset(val_s, tokenizer, code_vocab=code_vocab, scaler=train_ds.scaler, fit_scaler=False)
    test_ds = FinReportDataset(test_s, tokenizer, code_vocab=code_vocab, scaler=train_ds.scaler, fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if return_metadata:
        metadata = {
            "code_vocab": code_vocab,
            "n_codes": code_vocab.n_codes,
            "label_names": cfg.label_names,
            "label_mode": cfg.label_mode,
            "max_news_per_day": cfg.max_news_per_day,
            "news_pooling": cfg.news_pooling,
            "train_label_counts": {int(k): int(v) for k, v in pd.Series([s["label"] for s in train_s]).value_counts().to_dict().items()},
        }
        return train_loader, val_loader, test_loader, train_ds.scaler, metadata
    return train_loader, val_loader, test_loader, train_ds.scaler


if __name__ == "__main__":
    import sys

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw"
    train_loader, val_loader, test_loader, scaler, meta = run_preprocessing(
        raw_dir=raw_dir,
        lookback=cfg.lookback,
        batch_size=8,
        return_metadata=True,
    )
    batch = next(iter(train_loader))
    print("\n[Batch Shape Check]")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
    print(f"n_codes={meta['n_codes']}")
