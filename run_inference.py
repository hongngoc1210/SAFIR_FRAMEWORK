from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from src.config import Config as cfg
from src.data_preprocessing import CodeVocabulary, MarketStatusBuilder, NewsTokenizer, RawDataCleaner
from src.model_main import FinReportNextGen
from src.sep_report import generate_sep_report


warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinReport-NextGen v3 inference and report generation.")
    parser.add_argument("--data", type=str, required=True, help="Raw TSV/CSV file for inference.")
    parser.add_argument("--model", type=str, required=True, help="v3 checkpoint path.")

    parser.add_argument("--asset-dir", type=str, default="artifacts/preprocessing_v3")
    parser.add_argument("--scaler", type=str, default=None)
    parser.add_argument("--code-vocab", type=str, default=None)

    parser.add_argument("--bert-model", type=str, default=None)
    parser.add_argument("--max-text-len", type=int, default=None)
    parser.add_argument("--max-news-per-day", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--label-mode", type=str, default=None, choices=["binary", "ternary"])
    parser.add_argument("--news-pooling", type=str, default=None, choices=["cap", "sap", "pa_sap"])

    parser.add_argument("--code", type=str, default=None)
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run inference for all rows after optional --code/--date filtering.",
    )
    parser.add_argument("--batch", action="store_true", help="Run only the latest top-N rows, useful for testing.")
    parser.add_argument("--top-n", type=int, default=10)

    parser.add_argument("--output", type=str, default="./reports")
    parser.add_argument("--lang", type=str, default="both", choices=["en", "zh", "both"])
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Only save inference_summary.csv without generating PDF reports.",
    )

    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def configure_label_mode(label_mode: str) -> None:
    cfg.label_mode = label_mode
    if label_mode == "binary":
        cfg.n_classes = 2
        cfg.label_names = ("DOWN", "UP")
    else:
        cfg.n_classes = 3
        cfg.label_names = ("NEGATIVE", "NEUTRAL", "POSITIVE")


def configure_runtime(args: argparse.Namespace, meta: Optional[Dict] = None) -> None:
    if args.device is not None:
        cfg.device = args.device
    elif torch.cuda.is_available():
        cfg.device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cfg.device = "mps"
    else:
        cfg.device = "cpu"

    cfg.fp16 = bool(args.fp16 and cfg.device == "cuda")
    if meta:
        if "label_mode" in meta:
            configure_label_mode(str(meta["label_mode"]))
        cfg.lookback = int(meta.get("lookback", cfg.lookback))
        cfg.max_text_len = int(meta.get("max_text_len", cfg.max_text_len))
        cfg.max_news_per_day = int(meta.get("max_news_per_day", cfg.max_news_per_day))
        cfg.bert_model = str(meta.get("bert_model", cfg.bert_model))
        cfg.news_pooling = str(meta.get("news_pooling", cfg.news_pooling))

    if args.label_mode is not None:
        configure_label_mode(args.label_mode)
    if args.lookback is not None:
        cfg.lookback = int(args.lookback)
    if args.max_text_len is not None:
        cfg.max_text_len = int(args.max_text_len)
    if args.max_news_per_day is not None:
        cfg.max_news_per_day = int(args.max_news_per_day)
    if args.bert_model is not None:
        cfg.bert_model = args.bert_model
    if args.news_pooling is not None:
        cfg.news_pooling = args.news_pooling


def load_assets(args: argparse.Namespace):
    asset_dir = Path(args.asset_dir).expanduser().resolve()
    scaler_path = Path(args.scaler).expanduser().resolve() if args.scaler else asset_dir / "scaler.joblib"
    vocab_path = Path(args.code_vocab).expanduser().resolve() if args.code_vocab else asset_dir / "code_vocab.joblib"
    meta_path = asset_dir / "preprocessing_meta.joblib"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}. Run main.py first or pass --scaler.")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Code vocabulary not found: {vocab_path}. Run main.py first or pass --code-vocab.")

    scaler = joblib.load(scaler_path)
    code_vocab = joblib.load(vocab_path)
    meta = joblib.load(meta_path) if meta_path.exists() else {}
    print(f"[Assets] Loaded scaler     : {scaler_path}")
    print(f"[Assets] Loaded code_vocab : {vocab_path} (n_codes={code_vocab.n_codes})")
    if meta:
        print(f"[Assets] Loaded meta       : {meta_path}")
    return scaler, code_vocab, meta


def load_model(checkpoint_path: str, n_codes: int) -> FinReportNextGen:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    print(f"[Model] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=cfg.device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = FinReportNextGen(n_codes=n_codes, enable_sefn=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Model] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Model] Unexpected keys: {len(unexpected)}")
    if isinstance(ckpt, dict):
        print(f"[Model] checkpoint_info: epoch={ckpt.get('epoch', '?')} best_score={ckpt.get('best_score', '?')}")

    model.to(cfg.device)
    model.eval()
    print(f"[Model] Loaded successfully on {cfg.device}")
    return model


def load_and_prepare_df(data_path: str) -> pd.DataFrame:
    cleaner = RawDataCleaner(data_path)
    df = cleaner.load().drop_legacy_cols().normalize_date().fill_missing_text().get_df()
    df = MarketStatusBuilder(df).compute()
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def _safe_path_part(value: object) -> str:
    text = str(value) if value is not None else "UNKNOWN"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "UNKNOWN"


def _sort_rows_for_inference(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    sort_cols = [col for col in ["trade_date", "CODE"] if col in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols, ascending=ascending)


def select_rows(
    df: pd.DataFrame,
    code: Optional[str],
    date: Optional[str],
    full: bool,
    batch: bool,
    top_n: int,
) -> pd.DataFrame:
    """Select rows for inference.

    Modes:
      --full              : all rows after optional --code/--date filtering
      --batch --top-n N   : latest N rows, useful for testing
      default             : latest one row, same behavior as the original script
    """
    result = df.copy()

    if code:
        result = result[result["CODE"].astype(str).str.contains(str(code), na=False)]
        if result.empty:
            raise ValueError(f"No rows found for code: {code}")

    if date:
        result = result[result["trade_date"].astype(str).str.startswith(str(date))]
        if result.empty:
            raise ValueError(f"No rows found for date={date}, code={code}")

    if full:
        result = _sort_rows_for_inference(result, ascending=True)
    elif batch:
        result = _sort_rows_for_inference(result, ascending=False).head(top_n)
    else:
        result = _sort_rows_for_inference(result, ascending=True).tail(1)

    return result.reset_index(drop=True)


def extract_price_array(row: pd.Series, lookback: int) -> np.ndarray:
    records = []
    for i in range(1, 6):
        o = pd.to_numeric(row.get(f"open{i}", 0.0), errors="coerce")
        c = pd.to_numeric(row.get(f"close{i}", 0.0), errors="coerce")
        records.append([float(o) if pd.notna(o) else 0.0, float(c) if pd.notna(c) else 0.0])
    arr = np.asarray(records, dtype=np.float32)
    pad_len = lookback - len(arr)
    if pad_len > 0:
        arr = np.concatenate([np.zeros((pad_len, 2), dtype=np.float32), arr], axis=0)
    else:
        arr = arr[-lookback:]
    return arr


def encode_label(raw_label) -> Optional[int]:
    if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
        return None
    try:
        y = float(raw_label)
    except Exception:
        return None
    if cfg.label_mode == "binary":
        return 1 if y > 0 else 0
    if y < 0:
        return 0
    if y > 0:
        return 2
    return 1


def prepare_single_sample(row: pd.Series, scaler, tokenizer: NewsTokenizer, code_vocab: CodeVocabulary, lookback: int) -> Dict:
    news_text = str(row.get("text_a", "") or "")
    enc = tokenizer.encode_many(news_text)

    price_arr = extract_price_array(row, lookback=lookback)
    price_scaled = scaler.transform(price_arr).astype(np.float32)

    code = str(row.get("CODE", "UNKNOWN"))
    code_id = int(code_vocab.encode(code))
    if code_id == 0:
        print(f"[Warning] CODE={code} not found in training code_vocab; using UNK id=0.")

    mkt_mean = float(row.get("mkt_mean", 0.0) or 0.0)
    mkt_std = float(row.get("mkt_std", 0.0) or 0.0)
    raw_label = row.get("label", None)
    label = encode_label(raw_label)

    return {
        "input_ids": enc["input_ids"].unsqueeze(0).to(cfg.device),
        "attention_mask": enc["attention_mask"].unsqueeze(0).to(cfg.device),
        "news_item_mask": enc["news_item_mask"].unsqueeze(0).to(cfg.device),
        "code_ids": torch.tensor([code_id], dtype=torch.long, device=cfg.device),
        "price_seq": torch.tensor(price_scaled, dtype=torch.float32).unsqueeze(0).to(cfg.device),
        "mkt_vector": torch.tensor([[mkt_mean, mkt_std]], dtype=torch.float32, device=cfg.device),
        "meta": {
            "code": code,
            "date": str(row.get("DATE", row.get("trade_date", "")))[:10],
            "trade_date": str(row.get("trade_date", ""))[:10],
            "news_text": news_text,
            "label": label,
            "raw_label": raw_label,
        },
    }


@torch.no_grad()
def run_model_inference(model: FinReportNextGen, sample: Dict) -> Dict:
    with autocast(enabled=cfg.fp16 and cfg.device == "cuda"):
        out = model(
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            news_item_mask=sample["news_item_mask"],
            code_ids=sample["code_ids"],
            price_seq=sample["price_seq"],
            mkt_vector=sample["mkt_vector"],
            generate_text=False,
        )

    logits = out["logits"][0].float()
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])
    label_names = list(cfg.label_names)
    prediction = label_names[pred_id] if pred_id < len(label_names) else str(pred_id)

    qf = out["quant_factors"][0].float().detach().cpu()
    gate = out.get("gate_weights")
    gate_mean = float(gate[0].float().mean().item()) if isinstance(gate, torch.Tensor) else 0.0

    factor_names = [
        "Stock-aware News Factor",
        "Bidirectional News-Price Fusion Factor",
        "Market Gate Strength",
        "Causal Temporal Factor",
        "Risk Representation Factor",
    ]
    factors = {}
    chunk = max(1, len(qf) // len(factor_names))
    for i, name in enumerate(factor_names):
        if name == "Market Gate Strength":
            factors[name] = gate_mean
        else:
            start = i * chunk
            end = min(len(qf), start + chunk)
            raw = float(qf[start:end].mean().item()) if start < len(qf) else 0.0
            factors[name] = max(min(raw, 5.0), -5.0)

    if cfg.label_mode == "binary":
        direction = 1.0 if pred_id == 1 else -1.0
    else:
        direction = pred_id - ((cfg.n_classes - 1) / 2.0)
    ret_center = float(np.tanh(logits[pred_id].item()) * 0.01 * direction)
    var_est = float(out["var_est"][0].item())
    ret_low = ret_center - var_est
    ret_high = ret_center + var_est

    return {
        "prediction": prediction,
        "pred_id": pred_id,
        "confidence": confidence,
        "probabilities": {label_names[i]: float(p) for i, p in enumerate(probs) if i < len(label_names)},
        "var_estimate": var_est,
        "log_vol": float(out["log_vol"][0].item()),
        "factors": factors,
        "ret_low": ret_low,
        "ret_high": ret_high,
    }


def generate_report_for_sample(meta: Dict, result: Dict, output_dir: Path, languages: List[str]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if generate_sep_report is None:
        return {}
    paths = generate_sep_report(
        stock_code=meta["code"],
        date=meta["date"],
        news_text=meta["news_text"] or "(No news available for this date)",
        prediction=result["prediction"],
        confidence=result["confidence"],
        var_estimate=result["var_estimate"],
        factors=result["factors"],
        ret_low=result["ret_low"],
        ret_high=result["ret_high"],
        output_dir=str(output_dir),
        languages=languages,
    )
    return paths or {}


def save_summary_csv(records: List[Dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "inference_summary.csv"

    base_fields = [
        "sample_id", "code", "date", "prediction", "pred_id", "confidence", "var_estimate", "log_vol",
        "ret_low", "ret_high", "actual_label", "raw_label", "report_en", "report_zh",
    ]
    extra_fields = sorted(
        {
            key
            for record in records
            for key in record.keys()
            if key not in base_fields
        }
    )
    fieldnames = base_fields + extra_fields

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def main() -> List[Dict]:
    args = parse_args()
    scaler, code_vocab, meta = load_assets(args)
    configure_runtime(args, meta=meta)

    data_path = Path(args.data).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    languages = {"en": ["en"], "zh": ["zh"], "both": ["en", "zh"]}[args.lang]

    print("\n" + "═" * 55)
    print("  STEP 1 / 4 — Runtime")
    print("═" * 55)
    print(f"  Device       : {cfg.device}")
    print(f"  Label mode   : {cfg.label_mode} {cfg.label_names}")
    print(f"  News pooling : {cfg.news_pooling}")
    print(f"  Max news/day : {cfg.max_news_per_day}")

    print("\n" + "═" * 55)
    print("  STEP 2 / 4 — Load Data & Model")
    print("═" * 55)
    df = load_and_prepare_df(str(data_path))
    rows = select_rows(df, code=args.code, date=args.date, full=args.full, batch=args.batch, top_n=args.top_n)
    print(f"  Total rows after cleaning : {len(df):,}")
    print(f"  Rows selected            : {len(rows):,}")
    print(f"  Selection mode           : {'FULL' if args.full else ('BATCH' if args.batch else 'SINGLE')}")

    model = load_model(str(model_path), n_codes=code_vocab.n_codes)
    tokenizer = NewsTokenizer(model_name=cfg.bert_model, max_length=cfg.max_text_len, max_news=cfg.max_news_per_day)

    print("\n" + "═" * 55)
    print(f"  STEP 3 / 4 — Inference ({len(rows)} sample(s))")
    print("═" * 55)

    results: List[Dict] = []
    report_inputs = []
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        sample = prepare_single_sample(row=row, scaler=scaler, tokenizer=tokenizer, code_vocab=code_vocab, lookback=cfg.lookback)
        meta_row = sample["meta"]
        result = run_model_inference(model, sample)

        actual_label_name = ""
        if meta_row["label"] is not None and 0 <= int(meta_row["label"]) < len(cfg.label_names):
            actual_label_name = cfg.label_names[int(meta_row["label"])]

        print(f"\n  [{idx}/{len(rows)}] {meta_row['code']}  {meta_row['date']}")
        print(f"    Prediction : {result['prediction']}  (conf={result['confidence']:.0%})")
        print(f"    Probs      : {result['probabilities']}")
        print(f"    VaR        : {result['var_estimate']:.2%}")
        print(f"    Return est : {result['ret_low']*100:+.2f}% ~ {result['ret_high']*100:+.2f}%")
        if actual_label_name:
            match = "✓" if actual_label_name == result["prediction"] else "✗"
            print(f"    Actual     : {actual_label_name}  {match}")

        flat = {
            "sample_id": idx,
            "code": meta_row["code"],
            "date": meta_row["date"],
            "prediction": result["prediction"],
            "pred_id": result["pred_id"],
            "confidence": result["confidence"],
            "var_estimate": result["var_estimate"],
            "log_vol": result["log_vol"],
            "ret_low": result["ret_low"],
            "ret_high": result["ret_high"],
            "actual_label": actual_label_name,
            "raw_label": meta_row["raw_label"],
        }
        for label_name, prob in result.get("probabilities", {}).items():
            flat[f"prob_{label_name}"] = prob
        for factor_name, value in result.get("factors", {}).items():
            flat[f"factor_{_safe_path_part(factor_name)}"] = value

        results.append(flat)
        report_inputs.append((len(results) - 1, meta_row, result))

    print("\n" + "═" * 55)
    print("  STEP 4 / 4 — Save Reports / Summary")
    print("═" * 55)

    generated_count = 0
    should_generate_reports = (not args.no_report) and (generate_sep_report is not None)

    if should_generate_reports:
        multiple_samples = len(report_inputs) > 1
        for sample_idx, (record_idx, meta_row, result) in enumerate(report_inputs, start=1):
            print(f"\n  Generating report [{sample_idx}/{len(report_inputs)}]: {meta_row['code']}  {meta_row['date']}")

            # Use a unique folder per sample when generating multiple reports.
            # This avoids overwriting PDFs when the same stock/date appears in multiple rows.
            if multiple_samples:
                sample_dir = output_dir / f"{sample_idx:06d}_{_safe_path_part(meta_row['code'])}_{_safe_path_part(meta_row['date'])}"
            else:
                sample_dir = output_dir

            paths = generate_report_for_sample(meta_row, result, sample_dir, languages)
            for lang, path in paths.items():
                print(f"    [{lang.upper()}] {path}")
                results[record_idx][f"report_{lang}"] = path
                generated_count += 1
    elif args.no_report:
        print("  [Report] --no-report enabled; skipping PDF generation.")
    else:
        print("  [Report] src.sep_report.generate_sep_report not found; skipping PDF generation.")

    # Save CSV automatically in --full mode because it is the easiest way to audit all outputs.
    if args.save_csv or args.full or args.no_report or generate_sep_report is None:
        summary_path = save_summary_csv(results, output_dir)
        print(f"  [CSV] {summary_path}")

    print("\n" + "═" * 55)
    print(f"  Done. PDF files generated: {generated_count}. Output: {output_dir}")
    print("═" * 55)
    return results


if __name__ == "__main__":
    main()