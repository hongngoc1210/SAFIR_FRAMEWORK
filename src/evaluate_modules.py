from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
# from .config import Config as cfg
# from .model_main import FinReportNextGen
# from .data_preprocessing import run_preprocessing


# ---------------------------------------------------------------------------
# Robust package imports
# ---------------------------------------------------------------------------

def _import_package_modules():
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



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate intermediate outputs of FinReport-NextGen v3 modules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-path", type=str, required=True, help="Folder containing train.csv, val.csv, test.csv")
    parser.add_argument("--model", type=str, required=True, help="Path to best checkpoint, e.g. finreport_nextgen_v3_best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--output-dir", type=str, default="module_eval_outputs", help="Folder to save CSV/JSON outputs")

    parser.add_argument("--label-mode", type=str, default="binary", choices=["binary", "ternary"], help="Must match training setting")
    parser.add_argument("--news-pooling", type=str, default="sap", choices=["sap", "cap", "pa_sap"], help="Must match training setting")
    parser.add_argument("--max-news-per-day", type=int, default=4, help="Must match training setting used for v3 run")
    parser.add_argument("--max-text-len", type=int, default=128, help="BERT max token length")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback window")

    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Override device")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug option: only evaluate first N batches")
    parser.add_argument("--save-sample-csv", action="store_true", help="Save per-sample outputs to CSV")
    parser.add_argument("--save-embeddings", action="store_true", help="Save news_factor/quant_factors arrays as .npz")

    # Contribution / ablation modes. These do not retrain; they evaluate the same
    # checkpoint under controlled input removal.
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "zero_news", "zero_price", "zero_market"],
        help=(
            "Evaluate module contribution by zeroing a modality at inference. "
            "full = normal model, zero_news = DMAQ gets zero news_factor, "
            "zero_price = price_seq is zeroed, zero_market = mkt_vector is zeroed."
        ),
    )

    # Optional SEFN. This can download/load a causal LM, so keep off by default.
    parser.add_argument("--enable-sefn", action="store_true", help="Instantiate SEFN and generate a few explanations")
    parser.add_argument("--explain-samples", type=int, default=3, help="Number of SEFN explanations to save if --enable-sefn")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------------

def apply_cli_config(args: argparse.Namespace) -> None:
    if args.device is not None:
        cfg.device = args.device
    elif cfg.device == "cuda" and not torch.cuda.is_available():
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


def load_model(checkpoint_path: str, n_codes: int, enable_sefn: bool = False) -> torch.nn.Module:
    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(ckpt, dict) and "config" in ckpt:
        print(f"[Model] checkpoint config: {ckpt['config']}")

    model = FinReportNextGen(n_codes=n_codes, enable_sefn=enable_sefn)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Model] Missing keys: {len(missing)} key(s). Usually OK if SEFN setting differs.")
    if unexpected:
        print(f"[Model] Unexpected keys: {len(unexpected)} key(s). Usually OK if checkpoint has unused modules.")

    model.to(cfg.device)
    model.eval()
    return model


def select_loader(loaders: Tuple[Any, Any, Any], split: str):
    train_loader, val_loader, test_loader = loaders
    return {"train": train_loader, "val": val_loader, "test": test_loader}[split]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def tensor_stats(x: torch.Tensor, prefix: str) -> Dict[str, float]:
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_std": float(x.std(unbiased=False).item()),
        f"{prefix}_min": float(x.min().item()),
        f"{prefix}_max": float(x.max().item()),
    }


def norm_stats(x: torch.Tensor, prefix: str) -> Dict[str, float]:
    n = x.detach().float().norm(dim=-1).cpu()
    return tensor_stats(n, prefix)


def normalized_entropy(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return entropy normalized by log(sequence_length). weights: (B, S)."""
    w = weights.clamp_min(eps)
    ent = -(w * w.log()).sum(dim=-1)
    denom = math.log(max(weights.size(-1), 2))
    return ent / denom


def safe_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: Tuple[str, ...]) -> Dict[str, Any]:
    labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(label_names),
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_modules(model: torch.nn.Module, loader, args: argparse.Namespace) -> Dict[str, Any]:
    label_names = tuple(cfg.label_names)

    # Aggregates
    y_true: List[int] = []
    y_pred: List[int] = []
    probs_all: List[np.ndarray] = []

    ce_losses: List[float] = []
    log_vol_abs_errors: List[float] = []
    var_abs_errors: List[float] = []

    news_factor_norms: List[float] = []
    quant_factor_norms: List[float] = []
    valid_news_counts: List[int] = []
    attn_entropy_values: List[float] = []
    attn_max_values: List[float] = []
    sap_stock_attn_values: List[float] = []
    confidence_values: List[float] = []
    gate_entropy_values: List[float] = []
    gate_max_values: List[float] = []

    gate_collector: List[torch.Tensor] = []
    news_factor_collector: List[torch.Tensor] = []
    quant_factor_collector: List[torch.Tensor] = []

    rows: List[Dict[str, Any]] = []
    explanations: List[Dict[str, Any]] = []

    pbar = tqdm(loader, desc=f"Evaluate {args.split} [{args.ablation}]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        news_item_mask = batch["news_item_mask"].to(cfg.device)
        code_ids = batch["code_id"].to(cfg.device)
        price_seq = batch["price_seq"].to(cfg.device)
        mkt_vector = batch["mkt_vector"].to(cfg.device)
        labels = batch["label"].to(cfg.device)

        if args.ablation == "zero_price":
            price_seq = torch.zeros_like(price_seq)
        if args.ablation == "zero_market":
            mkt_vector = torch.zeros_like(mkt_vector)

        # -------------------------
        # Module I: ISF
        # -------------------------
        news_factor, attn_weights = model.isf(
            input_ids=input_ids,
            attention_mask=attention_mask,
            code_ids=code_ids,
            news_item_mask=news_item_mask,
            return_attn=True,
        )
        if args.ablation == "zero_news":
            news_factor = torch.zeros_like(news_factor)

        # Attention weights shape: (B, heads, query_len=1, key_len)
        attn = attn_weights.detach().float().mean(dim=1).squeeze(1)  # (B, key_len)
        attn_ent = normalized_entropy(attn)
        attn_max = attn.max(dim=-1).values
        attn_entropy_values.extend(attn_ent.cpu().tolist())
        attn_max_values.extend(attn_max.cpu().tolist())
        if cfg.news_pooling == "sap" and attn.size(-1) > 1:
            sap_stock_attn_values.extend(attn[:, 0].cpu().tolist())

        valid_counts = news_item_mask.sum(dim=-1).detach().cpu().tolist()
        valid_news_counts.extend([int(x) for x in valid_counts])
        news_norm = news_factor.detach().float().norm(dim=-1)
        news_factor_norms.extend(news_norm.cpu().tolist())

        # -------------------------
        # Module II: DMAQ
        # -------------------------
        dmaq_out = model.dmaq(
            price_seq=price_seq,
            mkt_vector=mkt_vector,
            news_factor=news_factor,
            code_ids=code_ids,
        )
        logits = dmaq_out["logits"]
        probs = F.softmax(logits.float(), dim=-1)
        preds = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values

        ce = F.cross_entropy(logits, labels, reduction="none")
        ce_losses.extend(ce.detach().cpu().tolist())

        log_vol_target = batch["log_vol_target"].to(cfg.device)
        var_target = batch["var_target"].to(cfg.device)
        log_vol_abs = (dmaq_out["log_vol"] - log_vol_target).abs()
        var_abs = (dmaq_out["var_est"] - var_target).abs()
        log_vol_abs_errors.extend(log_vol_abs.detach().cpu().tolist())
        var_abs_errors.extend(var_abs.detach().cpu().tolist())

        quant_norm = dmaq_out["quant_factors"].detach().float().norm(dim=-1)
        quant_factor_norms.extend(quant_norm.cpu().tolist())
        confidence_values.extend(conf.detach().cpu().tolist())

        gate = dmaq_out["gate_weights"].detach().float()
        # Gate is not a probability distribution; normalize for entropy only.
        gate_prob = gate.clamp_min(1e-12) / gate.clamp_min(1e-12).sum(dim=-1, keepdim=True)
        gate_ent = normalized_entropy(gate_prob)
        gate_entropy_values.extend(gate_ent.cpu().tolist())
        gate_max_values.extend(gate.max(dim=-1).values.cpu().tolist())

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        probs_all.extend(probs.detach().cpu().numpy())

        gate_collector.append(gate.cpu())
        if args.save_embeddings:
            news_factor_collector.append(news_factor.detach().cpu())
            quant_factor_collector.append(dmaq_out["quant_factors"].detach().cpu())

        codes = batch.get("code", [None] * labels.size(0))
        dates = batch.get("date", [None] * labels.size(0))
        if isinstance(codes, tuple):
            codes = list(codes)
        if isinstance(dates, tuple):
            dates = list(dates)

        batch_probs = probs.detach().cpu().numpy()
        for i in range(labels.size(0)):
            row = {
                "batch_idx": batch_idx,
                "row_in_batch": i,
                "code": codes[i] if isinstance(codes, (list, tuple)) else None,
                "date": dates[i] if isinstance(dates, (list, tuple)) else None,
                "label_id": int(labels[i].detach().cpu().item()),
                "label_name": label_names[int(labels[i].detach().cpu().item())],
                "pred_id": int(preds[i].detach().cpu().item()),
                "pred_name": label_names[int(preds[i].detach().cpu().item())],
                "correct": int(preds[i].item() == labels[i].item()),
                "confidence": float(conf[i].detach().cpu().item()),
                "ce_loss": float(ce[i].detach().cpu().item()),
                "valid_news_count": int(valid_counts[i]),
                "news_factor_norm": float(news_norm[i].detach().cpu().item()),
                "attn_entropy_norm": float(attn_ent[i].detach().cpu().item()),
                "attn_max": float(attn_max[i].detach().cpu().item()),
                "quant_factor_norm": float(quant_norm[i].detach().cpu().item()),
                "log_vol_pred": float(dmaq_out["log_vol"][i].detach().cpu().item()),
                "log_vol_target": float(log_vol_target[i].detach().cpu().item()),
                "log_vol_abs_error": float(log_vol_abs[i].detach().cpu().item()),
                "var_pred": float(dmaq_out["var_est"][i].detach().cpu().item()),
                "var_target": float(var_target[i].detach().cpu().item()),
                "var_abs_error": float(var_abs[i].detach().cpu().item()),
                "gate_entropy_norm": float(gate_ent[i].detach().cpu().item()),
                "gate_max": float(gate.max(dim=-1).values[i].detach().cpu().item()),
            }
            if cfg.news_pooling == "sap" and attn.size(-1) > 1:
                row["sap_stock_token_attention"] = float(attn[i, 0].detach().cpu().item())
            for c_idx, p in enumerate(batch_probs[i].tolist()):
                row[f"prob_{label_names[c_idx]}"] = float(p)
            rows.append(row)

        # -------------------------
        # Module III: SEFN optional sample explanations
        # -------------------------
        if args.enable_sefn and model.sefn is not None and len(explanations) < args.explain_samples:
            remaining = args.explain_samples - len(explanations)
            take = min(remaining, labels.size(0))
            text_list = model.sefn.generate_explanation(
                quant_factors=dmaq_out["quant_factors"][:take],
                pred_label=preds[:take],
                true_label=labels[:take],
                confidence=conf[:take],
                log_vol=dmaq_out["log_vol"][:take],
                var_est=dmaq_out["var_est"][:take],
            )
            for j, text in enumerate(text_list):
                explanations.append(
                    {
                        "code": codes[j] if isinstance(codes, (list, tuple)) else None,
                        "date": dates[j] if isinstance(dates, (list, tuple)) else None,
                        "label": label_names[int(labels[j].detach().cpu().item())],
                        "prediction": label_names[int(preds[j].detach().cpu().item())],
                        "confidence": float(conf[j].detach().cpu().item()),
                        "explanation": text,
                    }
                )

        pbar.set_postfix(
            ce=f"{np.mean(ce_losses):.4f}",
            acc=f"{accuracy_score(y_true, y_pred):.4f}" if y_true else "nan",
        )

    if not y_true:
        raise RuntimeError("No samples were evaluated. Check --split, --max-batches, and data path.")

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    metrics = class_metrics(y_true_np, y_pred_np, label_names=label_names)

    # Aggregate module stats
    gate_all = torch.cat(gate_collector, dim=0) if gate_collector else torch.empty(0)

    summary: Dict[str, Any] = {
        "config": {
            "split": args.split,
            "ablation": args.ablation,
            "label_mode": cfg.label_mode,
            "label_names": list(cfg.label_names),
            "news_pooling": cfg.news_pooling,
            "max_news_per_day": cfg.max_news_per_day,
            "use_bidirectional_fusion": cfg.use_bidirectional_fusion,
            "device": cfg.device,
        },
        "n_samples": int(len(y_true)),
        "classification": metrics,
        "module_1_isf": {
            "meaning": "Representation-quality diagnostics only; Module I does not directly predict labels.",
            "valid_news_count_mean": float(np.mean(valid_news_counts)),
            "valid_news_count_std": float(np.std(valid_news_counts)),
            "zero_news_ratio": float(np.mean(np.asarray(valid_news_counts) == 0)),
            "news_factor_norm_mean": float(np.mean(news_factor_norms)),
            "news_factor_norm_std": float(np.std(news_factor_norms)),
            "attention_entropy_norm_mean": float(np.mean(attn_entropy_values)),
            "attention_entropy_norm_std": float(np.std(attn_entropy_values)),
            "attention_max_mean": float(np.mean(attn_max_values)),
            "attention_max_std": float(np.std(attn_max_values)),
        },
        "module_2_dmaq": {
            "ce_loss_mean": float(np.mean(ce_losses)),
            "ce_loss_std": float(np.std(ce_losses)),
            "confidence_mean": float(np.mean(confidence_values)),
            "confidence_std": float(np.std(confidence_values)),
            "quant_factor_norm_mean": float(np.mean(quant_factor_norms)),
            "quant_factor_norm_std": float(np.std(quant_factor_norms)),
            "gate_entropy_norm_mean": float(np.mean(gate_entropy_values)),
            "gate_entropy_norm_std": float(np.std(gate_entropy_values)),
            "gate_max_mean": float(np.mean(gate_max_values)),
            "gate_max_std": float(np.std(gate_max_values)),
            "log_vol_mae": float(np.mean(log_vol_abs_errors)),
            "var_mae": float(np.mean(var_abs_errors)),
        },
        "module_3_sefn": {
            "enabled": bool(args.enable_sefn),
            "n_explanations_generated": int(len(explanations)),
            "note": "SEFN is usually a report layer in v3. It is not part of classification metrics unless explicitly enabled.",
            "sample_explanations": explanations,
        },
    }

    if sap_stock_attn_values:
        summary["module_1_isf"]["sap_stock_token_attention_mean"] = float(np.mean(sap_stock_attn_values))
        summary["module_1_isf"]["sap_stock_token_attention_std"] = float(np.std(sap_stock_attn_values))

    summary["module_2_dmaq"].update(tensor_stats(gate_all, "gate_weight"))

    embedding_payload = {}
    if args.save_embeddings:
        if news_factor_collector:
            embedding_payload["news_factor"] = torch.cat(news_factor_collector, dim=0).numpy()
        if quant_factor_collector:
            embedding_payload["quant_factors"] = torch.cat(quant_factor_collector, dim=0).numpy()
        if probs_all:
            embedding_payload["probs"] = np.asarray(probs_all, dtype=np.float32)

    return {
        "summary": summary,
        "rows": rows,
        "embeddings": embedding_payload,
    }


# ---------------------------------------------------------------------------
# Saving / printing
# ---------------------------------------------------------------------------

def save_outputs(result: Dict[str, Any], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_{args.ablation}_{cfg.news_pooling}"

    summary_path = output_dir / f"module_summary_{suffix}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)

    if args.save_sample_csv:
        csv_path = output_dir / f"module_sample_outputs_{suffix}.csv"
        pd.DataFrame(result["rows"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] sample CSV: {csv_path}")

    if args.save_embeddings and result["embeddings"]:
        npz_path = output_dir / f"module_embeddings_{suffix}.npz"
        np.savez_compressed(npz_path, **result["embeddings"])
        print(f"[Saved] embeddings: {npz_path}")

    print(f"[Saved] summary JSON: {summary_path}")


def print_console_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("MODULE EVALUATION SUMMARY")
    print("=" * 72)

    print("\n[Classification - full output from Module II]")
    clf = summary["classification"]
    print(f"Accuracy      : {clf['accuracy']:.6f}")
    print(f"F1 macro      : {clf['f1_macro']:.6f}")
    print(f"F1 weighted   : {clf['f1_weighted']:.6f}")
    print("Confusion matrix:")
    print(np.asarray(clf["confusion_matrix"]))
    print("\nClassification report:")
    print(clf["classification_report"])

    print("\n[Module I - ISF / news pooling]")
    m1 = summary["module_1_isf"]
    print(f"Valid news count mean       : {m1['valid_news_count_mean']:.4f}")
    print(f"Zero-news ratio             : {m1['zero_news_ratio']:.4f}")
    print(f"News factor norm mean       : {m1['news_factor_norm_mean']:.4f}")
    print(f"Attention entropy mean      : {m1['attention_entropy_norm_mean']:.4f}")
    print(f"Attention max mean          : {m1['attention_max_mean']:.4f}")
    if "sap_stock_token_attention_mean" in m1:
        print(f"SAP stock-token attn mean   : {m1['sap_stock_token_attention_mean']:.4f}")

    print("\n[Module II - DMAQ / fusion + prediction + risk]")
    m2 = summary["module_2_dmaq"]
    print(f"CE loss mean                : {m2['ce_loss_mean']:.6f}")
    print(f"Confidence mean             : {m2['confidence_mean']:.4f}")
    print(f"Quant factor norm mean      : {m2['quant_factor_norm_mean']:.4f}")
    print(f"Gate entropy mean           : {m2['gate_entropy_norm_mean']:.4f}")
    print(f"Gate max mean               : {m2['gate_max_mean']:.4f}")
    print(f"Risk log_vol MAE            : {m2['log_vol_mae']:.6f}")
    print(f"Risk VaR MAE                : {m2['var_mae']:.6f}")

    print("\n[Module III - SEFN]")
    m3 = summary["module_3_sefn"]
    print(f"Enabled                     : {m3['enabled']}")
    print(f"Generated explanations      : {m3['n_explanations_generated']}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_cli_config(args)

    data_path = Path(args.data_path).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print("=" * 72)
    print("FinReport-NextGen v3 | Module Output Evaluator")
    print("=" * 72)
    print(f"Data path        : {data_path}")
    print(f"Model checkpoint : {model_path}")
    print(f"Split            : {args.split}")
    print(f"Ablation         : {args.ablation}")
    print(f"Device           : {cfg.device}")
    print(f"Label mode       : {cfg.label_mode} {cfg.label_names}")
    print(f"News pooling     : {cfg.news_pooling}")
    print(f"Max news/day     : {cfg.max_news_per_day}")
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

    model = load_model(str(model_path), n_codes=int(metadata["n_codes"]), enable_sefn=args.enable_sefn)
    result = evaluate_modules(model=model, loader=loader, args=args)
    save_outputs(result=result, output_dir=output_dir, args=args)
    print_console_summary(result["summary"])


if __name__ == "__main__":
    main()