from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# Robust package imports
# ---------------------------------------------------------------------------

def _import_package_modules():
    """Import v3 modules whether this file is run as a module or as a script.

    This avoids: ImportError: attempted relative import with no known parent package.
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
    isf_mod = importlib.import_module(f"{base}.model_isf")
    dmaq_mod = importlib.import_module(f"{base}.model_dmaq")
    main_mod = importlib.import_module(f"{base}.model_main")
    sefn_mod = importlib.import_module(f"{base}.model_sefn")

    return (
        cfg_mod.Config,
        data_mod.run_preprocessing,
        isf_mod.ISFModule,
        dmaq_mod.DMAQModule,
        main_mod.FinReportNextGen,
        sefn_mod.SEFNModule,
    )


cfg, run_preprocessing, ISFModule, DMAQModule, FinReportNextGen, SEFNModule = _import_package_modules()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate FinReport-NextGen v3 modules separately.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-path", type=str, required=True, help="Folder containing train.csv, val.csv, test.csv")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["isf_news", "dmaq_factors", "full_news_factors", "sefn", "all"],
        help="Which module/stage to train/evaluate.",
    )
    parser.add_argument("--output-dir", type=str, default="module_stage_outputs", help="Folder for checkpoints/tables")
    parser.add_argument("--model-out", type=str, default=None, help="Optional checkpoint path for single-stage training")
    parser.add_argument("--full-checkpoint", type=str, default=None, help="Full model checkpoint for --stage sefn")

    parser.add_argument("--label-mode", type=str, default="binary", choices=["binary", "ternary"])
    parser.add_argument("--news-pooling", type=str, default="sap", choices=["sap", "cap", "pa_sap"])
    parser.add_argument("--max-news-per-day", type=int, default=4)
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None, help="Override cfg.lr")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override cfg.weight_decay")
    parser.add_argument("--dropout", type=float, default=None, help="Override cfg.dropout")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights in CE loss")
    parser.add_argument("--risk-loss-weight", type=float, default=None, help="Override cfg.risk_loss_weight")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug: use first N batches per epoch/eval")

    parser.add_argument("--explain-samples", type=int, default=8, help="Number of SEFN samples to generate")
    parser.add_argument("--save-sample-preds", action="store_true", help="Save per-sample test predictions CSV")

    return parser.parse_args()


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
    cfg.use_class_weights = not args.no_class_weights
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.risk_loss_weight is not None:
        cfg.risk_loss_weight = args.risk_loss_weight


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Stage-specific models
# ---------------------------------------------------------------------------

class ISFNewsOnlyModel(nn.Module):
    """Module I only: news -> news_factor -> label.

    This is an auxiliary supervised head used only to measure whether the ISF news
    representation has predictive signal by itself.
    """

    def __init__(self, n_codes: int):
        super().__init__()
        self.isf = ISFModule(n_codes=n_codes)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.news_factor_dim),
            nn.Linear(cfg.news_factor_dim, cfg.news_factor_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.news_factor_dim // 2, cfg.n_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        news_factor = self.isf(
            input_ids=batch["input_ids"].to(cfg.device),
            attention_mask=batch["attention_mask"].to(cfg.device),
            code_ids=batch["code_id"].to(cfg.device),
            news_item_mask=batch.get("news_item_mask").to(cfg.device),
        )
        logits = self.head(news_factor)
        return {"logits": logits, "news_factor": news_factor}


class DMAQFactorsOnlyModel(nn.Module):
    """Module II only with quantitative factors.

    DMAQ receives price_seq + market_vector. The news_factor is explicitly zeroed,
    so this stage measures the quantitative/factor branch without news.
    """

    def __init__(self, n_codes: int):
        super().__init__()
        self.dmaq = DMAQModule(n_codes=n_codes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        price_seq = batch["price_seq"].to(cfg.device)
        mkt_vector = batch["mkt_vector"].to(cfg.device)
        code_ids = batch["code_id"].to(cfg.device)
        zero_news = torch.zeros(price_seq.size(0), cfg.news_factor_dim, device=price_seq.device)
        return self.dmaq(
            price_seq=price_seq,
            mkt_vector=mkt_vector,
            news_factor=zero_news,
            code_ids=code_ids,
        )


class FullNewsFactorsModel(nn.Module):
    """Module I + Module II: news + quantitative factors."""

    def __init__(self, n_codes: int):
        super().__init__()
        self.model = FinReportNextGen(n_codes=n_codes, enable_sefn=False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            input_ids=batch["input_ids"].to(cfg.device),
            attention_mask=batch["attention_mask"].to(cfg.device),
            news_item_mask=batch.get("news_item_mask").to(cfg.device),
            code_ids=batch["code_id"].to(cfg.device),
            price_seq=batch["price_seq"].to(cfg.device),
            mkt_vector=batch["mkt_vector"].to(cfg.device),
            true_labels=batch["label"].to(cfg.device),
            generate_text=False,
        )


# ---------------------------------------------------------------------------
# Training / evaluation utilities
# ---------------------------------------------------------------------------

def make_model(stage: str, n_codes: int) -> nn.Module:
    if stage == "isf_news":
        return ISFNewsOnlyModel(n_codes=n_codes)
    if stage == "dmaq_factors":
        return DMAQFactorsOnlyModel(n_codes=n_codes)
    if stage == "full_news_factors":
        return FullNewsFactorsModel(n_codes=n_codes)
    raise ValueError(f"No trainable classification model for stage={stage}")


def class_weight_tensor(train_loader) -> Optional[torch.Tensor]:
    if not cfg.use_class_weights:
        return None
    counts = torch.zeros(cfg.n_classes, dtype=torch.float32)
    for batch in train_loader:
        y = batch["label"].view(-1)
        for c in range(cfg.n_classes):
            counts[c] += (y == c).sum().float()
    counts = counts.clamp_min(1.0)
    freq = counts / counts.sum()
    power = getattr(cfg, "class_weight_power", 0.5)
    weights = (1.0 / freq).pow(power)
    weights = weights / weights.mean()
    print("[ClassWeights]", {cfg.label_names[i]: float(weights[i].item()) for i in range(cfg.n_classes)})
    return weights.to(cfg.device)


def risk_loss_from_output(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "log_vol" not in out or "var_est" not in out:
        # ISF-only stage has no risk head.
        return out["logits"].new_tensor(0.0)
    log_vol_target = batch["log_vol_target"].to(cfg.device)
    var_target = batch["var_target"].to(cfg.device)
    vol_loss = F.smooth_l1_loss(out["log_vol"], log_vol_target)
    var_loss = F.smooth_l1_loss(out["var_est"], var_target)
    return cfg.risk_loss_weight * (cfg.vol_loss_weight * vol_loss + cfg.var_loss_weight * var_loss)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    labels = list(range(cfg.n_classes))
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
            target_names=list(cfg.label_names),
            zero_division=0,
        ),
    }


def update_sample_rows(rows: List[Dict[str, Any]], batch: Dict[str, Any], logits: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor) -> None:
    probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().tolist()
    preds_cpu = preds.detach().cpu().tolist()
    codes = batch.get("code", [None] * len(labels_cpu))
    dates = batch.get("date", [None] * len(labels_cpu))
    if isinstance(codes, tuple):
        codes = list(codes)
    if isinstance(dates, tuple):
        dates = list(dates)
    for i, y in enumerate(labels_cpu):
        row = {
            "code": codes[i] if isinstance(codes, list) else None,
            "date": dates[i] if isinstance(dates, list) else None,
            "label_id": int(y),
            "label_name": cfg.label_names[int(y)],
            "pred_id": int(preds_cpu[i]),
            "pred_name": cfg.label_names[int(preds_cpu[i])],
            "correct": int(preds_cpu[i] == y),
            "confidence": float(probs[i].max()),
        }
        for j, p in enumerate(probs[i].tolist()):
            row[f"prob_{cfg.label_names[j]}"] = float(p)
        rows.append(row)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    ce_loss_fn: nn.Module,
    stage: str,
    split_name: str = "val",
    max_batches: Optional[int] = None,
    collect_rows: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    losses: List[float] = []
    ce_losses: List[float] = []
    risk_losses: List[float] = []
    rows: List[Dict[str, Any]] = []

    pbar = tqdm(loader, desc=f"{stage} [{split_name}]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        labels = batch["label"].to(cfg.device)
        out = model(batch)
        logits = out["logits"]
        ce = ce_loss_fn(logits, labels)
        risk = risk_loss_from_output(out, batch)
        loss = ce + risk
        preds = logits.argmax(dim=-1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        losses.append(float(loss.item()))
        ce_losses.append(float(ce.item()))
        risk_losses.append(float(risk.item()))
        if collect_rows:
            update_sample_rows(rows, batch, logits, labels, preds)

        if y_true:
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}", f1=f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    if not y_true:
        raise RuntimeError(f"No samples evaluated for split={split_name}")
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "loss": float(np.mean(losses)),
            "ce_loss": float(np.mean(ce_losses)),
            "risk_loss": float(np.mean(risk_losses)),
        }
    )
    return metrics, rows


def train_one_stage(
    stage: str,
    train_loader,
    val_loader,
    test_loader,
    n_codes: int,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print(f"TRAIN STAGE: {stage}")
    print("=" * 72)

    model = make_model(stage, n_codes=n_codes).to(cfg.device)
    weights = class_weight_tensor(train_loader)
    ce_loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_score = -float("inf")
    best_state = None
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []
        train_ce: List[float] = []
        train_risk: List[float] = []
        train_y: List[int] = []
        train_p: List[int] = []

        pbar = tqdm(train_loader, desc=f"{stage} [train epoch {epoch}]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            labels = batch["label"].to(cfg.device)
            out = model(batch)
            logits = out["logits"]
            ce = ce_loss_fn(logits, labels)
            risk = risk_loss_from_output(out, batch)
            loss = ce + risk
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            train_y.extend(labels.detach().cpu().tolist())
            train_p.extend(preds.detach().cpu().tolist())
            train_losses.append(float(loss.item()))
            train_ce.append(float(ce.item()))
            train_risk.append(float(risk.item()))
            pbar.set_postfix(
                loss=f"{np.mean(train_losses):.4f}",
                f1=f"{f1_score(train_y, train_p, average='macro', zero_division=0):.4f}",
            )

        train_metrics = compute_metrics(train_y, train_p)
        train_metrics.update({"loss": float(np.mean(train_losses)), "ce_loss": float(np.mean(train_ce)), "risk_loss": float(np.mean(train_risk))})
        val_metrics, _ = evaluate_model(model, val_loader, ce_loss_fn, stage, split_name="val", max_batches=args.max_batches)

        val_score = float(val_metrics["f1_macro"])
        improved = val_score > best_score + args.min_delta
        if improved:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "best_val_f1_macro": best_score,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "label_mode": cfg.label_mode,
                        "label_names": cfg.label_names,
                        "news_pooling": cfg.news_pooling,
                        "max_news_per_day": cfg.max_news_per_day,
                        "n_codes": n_codes,
                    },
                },
                checkpoint_path,
            )
        else:
            no_improve += 1

        hist_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_risk_loss": train_metrics["risk_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_ce_loss": val_metrics["ce_loss"],
            "val_risk_loss": val_metrics["risk_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "improved": improved,
        }
        history.append(hist_row)

        print(
            f"[Epoch {epoch}] "
            f"Train loss={train_metrics['loss']:.4f} f1={train_metrics['f1_macro']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} f1={val_metrics['f1_macro']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"{'*best*' if improved else f'no_improve={no_improve}/{args.patience}'}"
        )

        if no_improve >= args.patience:
            print(f"[EarlyStopping] stage={stage} stopped at epoch {epoch}; best val_f1_macro={best_score:.6f}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})

    test_metrics, test_rows = evaluate_model(
        model,
        test_loader,
        ce_loss_fn,
        stage,
        split_name="test",
        max_batches=args.max_batches,
        collect_rows=args.save_sample_preds,
    )
    print(f"\n[{stage}] TEST Classification Report:\n{test_metrics['classification_report']}")

    result = {
        "stage": stage,
        "checkpoint": str(checkpoint_path),
        "best_val_f1_macro": best_score,
        "history": history,
        "test_metrics": test_metrics,
        "test_rows": test_rows,
    }

    # Free GPU when --stage all is used.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# SEFN report-stage evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sefn_stage(test_loader, n_codes: int, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    if not args.full_checkpoint:
        raise ValueError("--stage sefn requires --full-checkpoint pointing to a full_news_factors checkpoint")

    checkpoint = Path(args.full_checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Full checkpoint not found: {checkpoint}")

    print("\n" + "=" * 72)
    print("MODULE III - SEFN REPORT LAYER")
    print("=" * 72)
    print("Note: SEFN is a narrative/report layer. Without human/expert explanation labels, this stage generates samples but is not a supervised classification benchmark.")

    model = FinReportNextGen(n_codes=n_codes, enable_sefn=True).to(cfg.device)
    ckpt = torch.load(checkpoint, map_location=cfg.device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[SEFN] Missing keys: {len(missing)}. This is expected if the checkpoint was saved with enable_sefn=False.")
    if unexpected:
        print(f"[SEFN] Unexpected keys: {len(unexpected)}")
    model.eval()

    rows: List[Dict[str, Any]] = []
    for batch in test_loader:
        labels = batch["label"].to(cfg.device)
        out = model(
            input_ids=batch["input_ids"].to(cfg.device),
            attention_mask=batch["attention_mask"].to(cfg.device),
            news_item_mask=batch.get("news_item_mask").to(cfg.device),
            code_ids=batch["code_id"].to(cfg.device),
            price_seq=batch["price_seq"].to(cfg.device),
            mkt_vector=batch["mkt_vector"].to(cfg.device),
            true_labels=labels,
            generate_text=True,
        )
        preds = out["pred_labels"].detach().cpu().tolist()
        confs = out["confidence"].detach().cpu().tolist()
        explanations = out.get("explanations") or []
        codes = list(batch.get("code", [None] * len(preds)))
        dates = list(batch.get("date", [None] * len(preds)))
        label_cpu = labels.detach().cpu().tolist()
        for i, text in enumerate(explanations):
            rows.append(
                {
                    "code": codes[i],
                    "date": dates[i],
                    "label": cfg.label_names[int(label_cpu[i])],
                    "prediction": cfg.label_names[int(preds[i])],
                    "confidence": float(confs[i]),
                    "correct": int(preds[i] == label_cpu[i]),
                    "explanation": text,
                }
            )
            if len(rows) >= args.explain_samples:
                break
        if len(rows) >= args.explain_samples:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sefn_explanation_samples.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[SEFN] Saved explanation samples: {csv_path}")
    return {
        "stage": "sefn",
        "note": "Generated explanations only. Not a standalone supervised metric unless explanation labels/reward model exist.",
        "n_samples": len(rows),
        "samples_csv": str(csv_path),
    }


# ---------------------------------------------------------------------------
# Table export
# ---------------------------------------------------------------------------

def stage_display(stage: str) -> Tuple[str, str, str]:
    if stage == "isf_news":
        return "Module I - ISF", "Ours-ISF", "News"
    if stage == "dmaq_factors":
        return "Module II - DMAQ", "Ours-DMAQ", "Factors"
    if stage == "full_news_factors":
        return "Module I+II - ISF+DMAQ", "Ours-Full", "News+Factors"
    if stage == "sefn":
        return "Module III - SEFN", "Ours-SEFN", "Narrative"
    return stage, "Ours", "-"


def build_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for res in results:
        if "test_metrics" not in res:
            continue
        setting, method, inputs = stage_display(res["stage"])
        m = res["test_metrics"]
        rows.append(
            {
                "Setting": setting,
                "Method": method,
                "Input": inputs,
                "Accuracy": round(100.0 * m["accuracy"], 2),
                "Precision": round(100.0 * m["precision_macro"], 2),
                "Recall": round(100.0 * m["recall_macro"], 2),
                "F1": round(100.0 * m["f1_macro"], 2),
                "Weighted F1": round(100.0 * m["f1_weighted"], 2),
                "Best Val F1": round(100.0 * res.get("best_val_f1_macro", float("nan")), 2),
            }
        )
    return pd.DataFrame(rows)


def save_results(results: List[Dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary without huge per-sample rows unless requested.
    json_safe: List[Dict[str, Any]] = []
    for r in results:
        rc = dict(r)
        if not args.save_sample_preds:
            rc.pop("test_rows", None)
        json_safe.append(rc)
    json_path = output_dir / "stagewise_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe, f, ensure_ascii=False, indent=2)
    print(f"[Saved] JSON: {json_path}")

    table = build_table(results)
    if not table.empty:
        csv_path = output_dir / "stagewise_module_table.csv"
        md_path = output_dir / "stagewise_module_table.md"
        tex_path = output_dir / "stagewise_module_table.tex"
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        md_path.write_text(table.to_markdown(index=False), encoding="utf-8")
        tex_path.write_text(table.to_latex(index=False, escape=False), encoding="utf-8")
        print(f"[Saved] table CSV: {csv_path}")
        print(f"[Saved] table MD : {md_path}")
        print(f"[Saved] table TeX: {tex_path}")
        print("\n" + table.to_markdown(index=False))

    if args.save_sample_preds:
        for r in results:
            if r.get("test_rows"):
                sample_path = output_dir / f"sample_predictions_{r['stage']}.csv"
                pd.DataFrame(r["test_rows"]).to_csv(sample_path, index=False, encoding="utf-8-sig")
                print(f"[Saved] sample predictions: {sample_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_cli_config(args)
    set_seed(args.seed)

    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ckpt_dir = output_dir / "checkpoints"

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    print("=" * 72)
    print("FinReport-NextGen v3 | Stage-wise Module Training")
    print("=" * 72)
    print(f"Data path          : {data_path}")
    print(f"Stage              : {args.stage}")
    print(f"Device             : {cfg.device}")
    print(f"Label mode         : {cfg.label_mode} {cfg.label_names}")
    print(f"News pooling       : {cfg.news_pooling}")
    print(f"Class weights      : {cfg.use_class_weights}")
    print(f"Epochs             : {args.epochs}")
    print(f"LR                 : {cfg.lr}")
    print(f"Output dir         : {output_dir}")
    print("=" * 72)

    train_loader, val_loader, test_loader, scaler, metadata = run_preprocessing(
        raw_dir=str(data_path),
        bert_model=cfg.bert_model,
        lookback=cfg.lookback,
        max_text_len=cfg.max_text_len,
        batch_size=cfg.batch_size,
        return_metadata=True,
    )
    n_codes = int(metadata["n_codes"])

    results: List[Dict[str, Any]] = []

    if args.stage == "sefn":
        results.append(run_sefn_stage(test_loader, n_codes=n_codes, args=args, output_dir=output_dir))
        save_results(results, output_dir, args)
        return

    stages = ["isf_news", "dmaq_factors", "full_news_factors"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if args.model_out and args.stage != "all":
            checkpoint_path = Path(args.model_out).expanduser().resolve()
        else:
            checkpoint_path = ckpt_dir / f"{stage}_best.pt"
        res = train_one_stage(
            stage=stage,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            n_codes=n_codes,
            args=args,
            checkpoint_path=checkpoint_path,
        )
        results.append(res)

    save_results(results, output_dir, args)


if __name__ == "__main__":
    main()
