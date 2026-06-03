from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Robust package imports
# ---------------------------------------------------------------------------

def _import_package_modules():
    """Import project modules whether this file is run as a package module or as a script.

    Supported:
      1) python -m finreport_nextgen_v3.evaluate_ablation_table ...
      2) python evaluate_ablation_table.py ...

    Put this file in the same folder as config.py/model_main.py/data_preprocessing.py.
    """
    if __package__:
        base = __package__
    else:
        pkg_dir = Path(__file__).resolve().parent
        parent = pkg_dir.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        base = pkg_dir.name

    cfg_mod = importlib.import_module(f"{base}.config")
    data_mod = importlib.import_module(f"{base}.data_preprocessing")
    model_mod = importlib.import_module(f"{base}.model_main")
    return cfg_mod.Config, data_mod.run_preprocessing, model_mod.FinReportNextGen


cfg, run_preprocessing, FinReportNextGen = _import_package_modules()


@dataclass(frozen=True)
class Variant:
    key: str
    setting: str
    input_type: str
    use_news: bool
    use_price: bool
    use_market: bool


VARIANTS: Dict[str, Variant] = {
    # Closest to FinReport-style ablation table.
    "factors_only": Variant(
        key="factors_only",
        setting="Factors only",
        input_type="Factors",
        use_news=False,
        use_price=True,
        use_market=True,
    ),
    "news_only": Variant(
        key="news_only",
        setting="News only",
        input_type="News",
        use_news=True,
        use_price=False,
        use_market=False,
    ),
    "news_factors": Variant(
        key="news_factors",
        setting="News + Factors",
        input_type="News+Factors",
        use_news=True,
        use_price=True,
        use_market=True,
    ),
    # Extra diagnostics. Not included by default.
    "price_only": Variant(
        key="price_only",
        setting="Price only",
        input_type="Price",
        use_news=False,
        use_price=True,
        use_market=False,
    ),
    "news_price": Variant(
        key="news_price",
        setting="News + Price",
        input_type="News+Price",
        use_news=True,
        use_price=True,
        use_market=False,
    ),
    "market_only": Variant(
        key="market_only",
        setting="Market only",
        input_type="Market",
        use_news=False,
        use_price=False,
        use_market=True,
    ),
    "full": Variant(
        key="full",
        setting="Full model",
        input_type="News+Factors+Market",
        use_news=True,
        use_price=True,
        use_market=True,
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a paper-style ablation comparison table for FinReport-NextGen v3. "
            "By default, this performs inference-time ablations on the same checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-path", type=str, required=True, help="Folder containing train.csv, val.csv, test.csv")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        action="append",
        help=(
            "Checkpoint path, or name=checkpoint_path. Can be repeated. "
            "Example: --model Ours=finreport_nextgen_v3_best.pt"
        ),
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--output-dir", type=str, default="ablation_table_outputs", help="Output folder")

    parser.add_argument("--label-mode", type=str, default="binary", choices=["binary", "ternary"], help="Must match training")
    parser.add_argument("--news-pooling", type=str, default="sap", choices=["sap", "cap", "pa_sap"], help="Must match training")
    parser.add_argument("--max-news-per-day", type=int, default=4, help="Must match training")
    parser.add_argument("--max-text-len", type=int, default=128, help="BERT max token length")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback window")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Override device")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug: evaluate first N batches only")

    parser.add_argument(
        "--variants",
        type=str,
        default="factors_only,news_only,news_factors",
        help=(
            "Comma-separated variants. Available: " + ",".join(VARIANTS.keys()) + ". "
            "Default mimics FinReport ablation rows."
        ),
    )
    parser.add_argument(
        "--average",
        type=str,
        default="macro",
        choices=["macro", "weighted", "micro"],
        help="Averaging for Precision/Recall/F1 columns",
    )
    parser.add_argument("--decimals", type=int, default=2, help="Number of decimals for percentage metrics")
    parser.add_argument("--include-loss", action="store_true", help="Also include CE loss column")
    parser.add_argument("--no-percent", action="store_true", help="Do not multiply metrics by 100")

    return parser.parse_args()


def apply_cli_config(args: argparse.Namespace) -> None:
    if args.device is not None:
        cfg.device = args.device
    elif getattr(cfg, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    cfg.label_mode = args.label_mode
    if args.label_mode == "binary":
        cfg.n_classes = 2
        cfg.label_names = ("DOWN", "UP")
    else:
        cfg.n_classes = 3
        cfg.label_names = ("NEGATIVE", "NEUTRAL", "POSITIVE")

    cfg.news_pooling = args.news_pooling
    cfg.max_news_per_day = args.max_news_per_day
    cfg.max_text_len = args.max_text_len
    cfg.lookback = args.lookback
    cfg.batch_size = args.batch_size


def parse_model_specs(model_specs: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for idx, spec in enumerate(model_specs):
        if "=" in spec:
            name, path = spec.split("=", 1)
            name = name.strip() or f"Model{idx + 1}"
        else:
            name, path = ("Ours" if len(model_specs) == 1 else f"Model{idx + 1}"), spec
        parsed.append((name, Path(path).expanduser().resolve()))
    return parsed


def parse_variants(variants_str: str) -> List[Variant]:
    keys = [x.strip() for x in variants_str.split(",") if x.strip()]
    unknown = [k for k in keys if k not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}. Available: {sorted(VARIANTS)}")
    return [VARIANTS[k] for k in keys]


# ---------------------------------------------------------------------------
# Model / data helpers
# ---------------------------------------------------------------------------

def select_loader(loaders: Tuple[Any, Any, Any], split: str):
    train_loader, val_loader, test_loader = loaders
    return {"train": train_loader, "val": val_loader, "test": test_loader}[split]


def load_model(checkpoint_path: Path, n_codes: int) -> torch.nn.Module:
    ckpt = torch.load(str(checkpoint_path), map_location=cfg.device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = FinReportNextGen(n_codes=n_codes, enable_sefn=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Model] Missing keys for {checkpoint_path.name}: {len(missing)}")
    if unexpected:
        print(f"[Model] Unexpected keys for {checkpoint_path.name}: {len(unexpected)}")
    model.to(cfg.device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_variant(model: torch.nn.Module, loader, variant: Variant, args: argparse.Namespace) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred: List[int] = []
    ce_losses: List[float] = []

    iterator = tqdm(loader, desc=variant.setting, leave=False)
    for batch_idx, batch in enumerate(iterator):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        news_item_mask = batch.get("news_item_mask")
        if news_item_mask is not None:
            news_item_mask = news_item_mask.to(cfg.device)
        code_ids = batch["code_id"].to(cfg.device)
        price_seq = batch["price_seq"].to(cfg.device)
        mkt_vector = batch["mkt_vector"].to(cfg.device)
        labels = batch["label"].to(cfg.device)

        if not variant.use_price:
            price_seq = torch.zeros_like(price_seq)
        if not variant.use_market:
            mkt_vector = torch.zeros_like(mkt_vector)

        if variant.use_news:
            news_factor = model.isf(
                input_ids=input_ids,
                attention_mask=attention_mask,
                code_ids=code_ids,
                news_item_mask=news_item_mask,
            )
        else:
            news_factor = torch.zeros((labels.size(0), cfg.news_factor_dim), dtype=price_seq.dtype, device=cfg.device)

        out = model.dmaq(
            price_seq=price_seq,
            mkt_vector=mkt_vector,
            news_factor=news_factor,
            code_ids=code_ids,
        )
        logits = out["logits"]
        preds = logits.argmax(dim=-1)
        ce = F.cross_entropy(logits, labels, reduction="none")

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        ce_losses.extend(ce.detach().cpu().tolist())

        if y_true:
            iterator.set_postfix(acc=f"{accuracy_score(y_true, y_pred):.4f}")

    if not y_true:
        raise RuntimeError("No samples evaluated. Check data split or --max-batches.")

    labels_all = list(range(cfg.n_classes))
    avg = args.average
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels_all, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels_all, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels_all, average=avg, zero_division=0)),
        "loss": float(np.mean(ce_losses)),
        "support": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_all).tolist(),
    }
    return metrics


def build_row(model_name: str, variant: Variant, metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    scale = 1.0 if args.no_percent else 100.0
    row = {
        "Setting": variant.setting,
        "Method": model_name,
        "Input": variant.input_type,
        "Accuracy": round(metrics["accuracy"] * scale, args.decimals),
        "Precision": round(metrics["precision"] * scale, args.decimals),
        "Recall": round(metrics["recall"] * scale, args.decimals),
        "F1": round(metrics["f1"] * scale, args.decimals),
    }
    if args.include_loss:
        row["Loss"] = round(metrics["loss"], 6)
    return row


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = df.astype(str).values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(row) for row in rows])


def save_outputs(table_df: pd.DataFrame, raw_results: Dict[str, Any], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_{cfg.news_pooling}_{args.average}"

    csv_path = out_dir / f"ablation_table_{suffix}.csv"
    md_path = out_dir / f"ablation_table_{suffix}.md"
    json_path = out_dir / f"ablation_raw_results_{suffix}.json"
    tex_path = out_dir / f"ablation_table_{suffix}.tex"

    table_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_text = to_markdown_table(table_df)
    md_path.write_text(md_text + "\n", encoding="utf-8")
    table_df.to_latex(tex_path, index=False, escape=False)
    json_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Saved] CSV : {csv_path}")
    print(f"[Saved] MD  : {md_path}")
    print(f"[Saved] TEX : {tex_path}")
    print(f"[Saved] JSON: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_cli_config(args)

    data_path = Path(args.data_path).expanduser().resolve()
    model_specs = parse_model_specs(args.model)
    variants = parse_variants(args.variants)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    for _, model_path in model_specs:
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print("=" * 72)
    print("FinReport-NextGen v3 | Paper-style Ablation Table")
    print("=" * 72)
    print(f"Data path      : {data_path}")
    print(f"Split          : {args.split}")
    print(f"Device         : {cfg.device}")
    print(f"Label mode     : {cfg.label_mode} {cfg.label_names}")
    print(f"News pooling   : {cfg.news_pooling}")
    print(f"Variants       : {', '.join(v.key for v in variants)}")
    print(f"Metric average : {args.average}")
    print("Note           : default mode is inference-time ablation on trained checkpoint(s).")
    print("=" * 72)

    train_loader, val_loader, test_loader, scaler, metadata = run_preprocessing(
        raw_dir=str(data_path),
        bert_model=cfg.bert_model,
        lookback=cfg.lookback,
        max_text_len=cfg.max_text_len,
        batch_size=cfg.batch_size,
        return_metadata=True,
    )
    loader = select_loader((train_loader, val_loader, test_loader), args.split)
    n_codes = int(metadata["n_codes"])

    table_rows: List[Dict[str, Any]] = []
    raw_results: Dict[str, Any] = {
        "config": {
            "split": args.split,
            "label_mode": cfg.label_mode,
            "label_names": list(cfg.label_names),
            "news_pooling": cfg.news_pooling,
            "max_news_per_day": cfg.max_news_per_day,
            "average": args.average,
            "percent": not args.no_percent,
            "variants": [v.key for v in variants],
            "note": "Inference-time ablation. For strict paper-style ablation, retrain separate checkpoints per variant.",
        },
        "results": {},
    }

    for model_name, model_path in model_specs:
        print(f"\n[Model] {model_name}: {model_path}")
        model = load_model(model_path, n_codes=n_codes)
        raw_results["results"][model_name] = {}

        for variant in variants:
            metrics = evaluate_variant(model, loader, variant, args)
            raw_results["results"][model_name][variant.key] = metrics
            table_rows.append(build_row(model_name, variant, metrics, args))

        # release if multiple models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table_df = pd.DataFrame(table_rows)
    print("\n" + to_markdown_table(table_df))
    save_outputs(table_df, raw_results, args)

    print("\nInterpretation tip:")
    print("- If News + Factors > Factors only, the news branch contributes useful signal.")
    print("- If Factors only ≈ News + Factors, text_a/news may be weak, noisy, or already reflected in price.")
    print("- For publication-level ablation, retrain separate models with modules disabled, not only zeroed at inference.")


if __name__ == "__main__":
    main()
