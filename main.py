from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch

from src.config import Config as cfg
from src.data_preprocessing import run_preprocessing
from src.model_main import FinReportNextGen
from src.trainer import FinReportTrainer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinReport-NextGen v3 training / evaluation entrypoint.")
    parser.add_argument("--data-path", type=str, required=True, help="Directory containing train.csv, val.csv, test.csv.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--label-mode", type=str, default=cfg.label_mode, choices=["binary", "ternary"])
    parser.add_argument("--news-pooling", type=str, default=cfg.news_pooling, choices=["cap", "sap", "pa_sap"])
    parser.add_argument("--max-news-per-day", type=int, default=cfg.max_news_per_day)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency CE class weights.")
    parser.add_argument("--no-bidirectional-fusion", action="store_true", help="Disable price-to-news/news-to-price attention.")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoint_v3.pt")
    parser.add_argument("--best-model-path", type=str, default="finreport_nextgen_v3_best.pt")

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--es-metric",
        type=str,
        default="val_f1_macro",
        choices=[
            "val_accuracy",
            "val_f1_macro",
            "val_f1_weighted",
            "val_precision_macro",
            "val_recall_macro",
            "val_loss",
        ],
    )

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--asset-dir", type=str, default="artifacts/preprocessing_v3")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(args: argparse.Namespace) -> None:
    if args.device is not None:
        cfg.device = args.device
    elif args.force_cpu:
        cfg.device = "cpu"
    elif torch.cuda.is_available():
        cfg.device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cfg.device = "mps"
    else:
        cfg.device = "cpu"

    cfg.fp16 = bool(args.fp16 and cfg.device == "cuda")
    cfg.batch_size = int(args.batch_size)
    cfg.news_pooling = args.news_pooling
    cfg.max_news_per_day = int(args.max_news_per_day)
    cfg.use_class_weights = not args.no_class_weights
    cfg.use_bidirectional_fusion = not args.no_bidirectional_fusion

    cfg.label_mode = args.label_mode
    if cfg.label_mode == "binary":
        cfg.n_classes = 2
        cfg.label_names = ("DOWN", "UP")
    else:
        cfg.n_classes = 3
        cfg.label_names = ("NEGATIVE", "NEUTRAL", "POSITIVE")


def save_preprocessing_assets(asset_dir: Path, scaler, metadata: Dict) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, asset_dir / "scaler.joblib")
    joblib.dump(metadata["code_vocab"], asset_dir / "code_vocab.joblib")
    joblib.dump(
        {
            "architecture": "FinReport-NextGen-v3",
            "n_codes": metadata["n_codes"],
            "label_names": tuple(metadata.get("label_names", cfg.label_names)),
            "label_mode": cfg.label_mode,
            "lookback": cfg.lookback,
            "max_text_len": cfg.max_text_len,
            "max_news_per_day": cfg.max_news_per_day,
            "bert_model": cfg.bert_model,
            "news_pooling": cfg.news_pooling,
            "use_bidirectional_fusion": cfg.use_bidirectional_fusion,
        },
        asset_dir / "preprocessing_meta.joblib",
    )
    print(f"[Assets] Saved scaler/code_vocab/meta to: {asset_dir}")


def load_model_weights(model: FinReportNextGen, checkpoint_path: str) -> Dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=cfg.device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys: {len(unexpected)}")
    if isinstance(ckpt, dict):
        print(f"[Checkpoint] Loaded {path} | epoch={ckpt.get('epoch', '?')} | best_score={ckpt.get('best_score', '?')}")
    return ckpt if isinstance(ckpt, dict) else {}


def build_data_and_model(data_path: Path, batch_size: int) -> Tuple:
    train_loader, val_loader, test_loader, scaler, metadata = run_preprocessing(
        raw_dir=str(data_path),
        bert_model=cfg.bert_model,
        lookback=cfg.lookback,
        max_text_len=cfg.max_text_len,
        batch_size=batch_size,
        return_metadata=True,
    )
    model = FinReportNextGen(n_codes=metadata["n_codes"], enable_sefn=False)
    trainer = FinReportTrainer(model)
    return train_loader, val_loader, test_loader, scaler, metadata, model, trainer


def final_evaluation(trainer: FinReportTrainer, test_loader) -> Dict[str, float]:
    print("\n" + "═" * 60)
    print("  Final Evaluation on Test Set")
    print("═" * 60)
    metrics = trainer.evaluate(test_loader, desc="Test Set [Best Model]", prefix="test", print_report=True)

    print("\n" + "─" * 60)
    print("  FINAL TEST METRICS")
    print("─" * 60)
    for label, key in [
        ("Accuracy", "test_accuracy"),
        ("Precision macro", "test_precision_macro"),
        ("Recall macro", "test_recall_macro"),
        ("F1 macro", "test_f1_macro"),
        ("Precision weighted", "test_precision_weighted"),
        ("Recall weighted", "test_recall_weighted"),
        ("F1 weighted", "test_f1_weighted"),
        ("CE loss", "test_ce_loss"),
        ("Risk loss", "test_risk_loss"),
        ("Total loss", "test_loss"),
    ]:
        print(f"  {label:<20}: {metrics.get(key, float('nan')):.6f}")
    print("─" * 60)
    return metrics


def main() -> Dict[str, float] | None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    asset_dir = Path(args.asset_dir).expanduser().resolve()

    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {data_path}")
    if args.epochs <= 0 and not args.eval_only:
        raise ValueError("--epochs must be positive unless --eval-only is used")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")

    set_seed(args.seed)
    configure_runtime(args)

    resume_training = bool(args.resume and not args.no_resume)
    patience = args.patience if args.patience > 0 else int(1e9)
    es_mode = "min" if args.es_metric == "val_loss" else "max"

    print("=" * 60)
    print("FinReport-NextGen v3")
    print("=" * 60)
    print(f"Device                  : {cfg.device}")
    print(f"FP16                    : {cfg.fp16}")
    print(f"Data path               : {data_path}")
    print(f"Epochs                  : {args.epochs}")
    print(f"Batch size              : {args.batch_size}")
    print(f"Seed                    : {args.seed}")
    print(f"Label mode              : {cfg.label_mode} {cfg.label_names}")
    print(f"News pooling            : {cfg.news_pooling}")
    print(f"Max news/day            : {cfg.max_news_per_day}")
    print(f"Bidirectional fusion    : {cfg.use_bidirectional_fusion}")
    print(f"Class weights           : {cfg.use_class_weights}")
    print(f"Resume                  : {resume_training}")
    print(f"Checkpoint              : {args.checkpoint_path}")
    print(f"Best model              : {args.best_model_path}")
    print(f"ES metric               : {args.es_metric} ({es_mode})")
    print(f"Asset dir               : {asset_dir}")
    print(f"Eval only               : {args.eval_only}")
    print("=" * 60)

    train_loader, val_loader, test_loader, scaler, metadata, model, trainer = build_data_and_model(
        data_path=data_path,
        batch_size=args.batch_size,
    )
    save_preprocessing_assets(asset_dir, scaler, metadata)

    if args.eval_only:
        load_model_weights(model, args.best_model_path)
        return final_evaluation(trainer, test_loader)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        resume=resume_training,
        checkpoint_path=args.checkpoint_path,
        patience=patience,
        min_delta=args.min_delta,
        es_metric=args.es_metric,
        es_mode=es_mode,
    )

    generated_best = Path("finreport_nextgen_v3_best.pt")
    requested_best = Path(args.best_model_path)
    if generated_best.exists() and generated_best.resolve() != requested_best.resolve():
        requested_best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_best, requested_best)
        print(f"[Checkpoint] Copied best model to: {requested_best}")

    best_path = str(requested_best if requested_best.exists() else generated_best)
    load_model_weights(model, best_path)
    return final_evaluation(trainer, test_loader)


if __name__ == "__main__":
    main()
