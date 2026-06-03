from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import Config as cfg
from .model_main import FinReportNextGen
from .model_sefn import PPOTrainer


class MetricsTracker:
    def __init__(self):
        self._preds: List[int] = []
        self._labels: List[int] = []
        self._loss_sum = 0.0
        self._loss_steps = 0
        self._ce_sum = 0.0
        self._risk_sum = 0.0

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None,
        ce_loss: Optional[float] = None,
        risk_loss: Optional[float] = None,
    ) -> None:
        self._preds.extend(preds.detach().cpu().tolist())
        self._labels.extend(labels.detach().cpu().tolist())
        if loss is not None:
            self._loss_sum += float(loss)
            self._loss_steps += 1
        if ce_loss is not None:
            self._ce_sum += float(ce_loss)
        if risk_loss is not None:
            self._risk_sum += float(risk_loss)

    def compute(self, prefix: str = "val") -> Dict[str, float]:
        if not self._labels:
            return {}
        y_true = np.array(self._labels)
        y_pred = np.array(self._preds)
        labels = list(range(cfg.n_classes))
        steps = max(1, self._loss_steps)
        return {
            f"{prefix}_loss": float(self._loss_sum / steps),
            f"{prefix}_ce_loss": float(self._ce_sum / steps),
            f"{prefix}_risk_loss": float(self._risk_sum / steps),
            f"{prefix}_accuracy": float((y_true == y_pred).mean()),
            f"{prefix}_precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            f"{prefix}_recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            f"{prefix}_f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            f"{prefix}_precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            f"{prefix}_recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            f"{prefix}_f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        }

    def classification_report(self) -> str:
        if not self._labels:
            return ""
        return classification_report(
            self._labels,
            self._preds,
            labels=list(range(cfg.n_classes)),
            target_names=list(cfg.label_names),
            zero_division=0,
        )


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max", restore_best: bool = True):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.triggered = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, model: nn.Module) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [EarlyStopping] improved -> {score:.6f}")
            return False

        self.counter += 1
        print(f"  [EarlyStopping] no improvement ({self.counter}/{self.patience}), best={self.best_score:.6f}")
        if self.counter >= self.patience:
            self.triggered = True
            if self.restore_best and self.best_state is not None:
                model.load_state_dict({k: v.to(cfg.device) for k, v in self.best_state.items()})
                print("  [EarlyStopping] best weights restored.")
            return True
        return False


class FinReportTrainer:
    """Trainer v3.

    Major changes:
    - Accepts v3 daily-news tensors and passes news_item_mask into the model.
    - Uses optional sqrt inverse-frequency class weights for imbalanced DOWN/UP labels.
    - Logs CE and risk losses separately.
    - Keeps PPO disabled unless real explanation IDs and a reward model are added later.
    """

    def __init__(self, model: FinReportNextGen):
        self.model = model.to(cfg.device)
        self.scaler = GradScaler(enabled=cfg.fp16)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.class_weights: Optional[torch.Tensor] = None
        self.ce_loss = nn.CrossEntropyLoss()
        self.ppo_trainer = PPOTrainer(model.sefn, lr=1e-5) if (cfg.enable_ppo and model.sefn is not None) else None

    def set_class_weights_from_loader(self, loader) -> None:
        if not cfg.use_class_weights:
            self.class_weights = None
            self.ce_loss = nn.CrossEntropyLoss()
            return

        counts = torch.zeros(cfg.n_classes, dtype=torch.float32)
        for batch in loader:
            labels = batch["label"].view(-1)
            for c in range(cfg.n_classes):
                counts[c] += (labels == c).sum().float()
        counts = counts.clamp_min(1.0)
        freq = counts / counts.sum()
        weights = (1.0 / freq).pow(cfg.class_weight_power)
        weights = weights / weights.mean()
        self.class_weights = weights.to(cfg.device)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        named = {cfg.label_names[i]: float(weights[i].item()) for i in range(cfg.n_classes)}
        print(f"[ClassWeights] enabled: {named}")

    def _risk_loss(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "log_vol_target" not in batch or "var_target" not in batch:
            return out["log_vol"].new_tensor(0.0)
        log_vol_target = batch["log_vol_target"].to(cfg.device)
        var_target = batch["var_target"].to(cfg.device)
        vol_loss = F.smooth_l1_loss(out["log_vol"], log_vol_target)
        var_loss = F.smooth_l1_loss(out["var_est"], var_target)
        return cfg.risk_loss_weight * (cfg.vol_loss_weight * vol_loss + cfg.var_loss_weight * var_loss)

    def _step_batch(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(cfg.device)
        attn_mask = batch["attention_mask"].to(cfg.device)
        news_item_mask = batch.get("news_item_mask")
        if news_item_mask is not None:
            news_item_mask = news_item_mask.to(cfg.device)
        code_ids = batch["code_id"].to(cfg.device)
        price_seq = batch["price_seq"].to(cfg.device)
        mkt_vec = batch["mkt_vector"].to(cfg.device)
        labels = batch["label"].to(cfg.device)

        with autocast(enabled=cfg.fp16):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                news_item_mask=news_item_mask,
                code_ids=code_ids,
                price_seq=price_seq,
                mkt_vector=mkt_vec,
                true_labels=labels,
                generate_text=False,
            )
            ce = self.ce_loss(out["logits"], labels)
            risk = self._risk_loss(out, batch)
            loss = ce + risk

        out["loss"] = loss
        out["ce_loss"] = ce.detach()
        out["risk_loss"] = risk.detach()
        return out

    def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        tracker = MetricsTracker()
        skipped = 0
        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch} [Train]", leave=False)

        for batch in pbar:
            out = self._step_batch(batch, train=True)
            loss = out["loss"]
            if not torch.isfinite(loss) or not torch.isfinite(out["logits"]).all():
                skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                if skipped <= 5:
                    print(f"[Train Warning] skipped non-finite batch. skipped={skipped}")
                continue

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            labels = batch["label"].to(cfg.device)
            preds = out["logits"].argmax(dim=-1)
            tracker.update(
                preds,
                labels,
                loss=float(loss.item()),
                ce_loss=float(out["ce_loss"].item()),
                risk_loss=float(out["risk_loss"].item()),
            )
            m = tracker.compute("train")
            if m:
                pbar.set_postfix(
                    loss=f"{m['train_loss']:.4f}",
                    ce=f"{m['train_ce_loss']:.4f}",
                    risk=f"{m['train_risk_loss']:.4f}",
                    acc=f"{m['train_accuracy']:.4f}",
                    f1=f"{m['train_f1_macro']:.4f}",
                )

        metrics = tracker.compute("train")
        metrics["train_skipped_batches"] = float(skipped)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader, desc: str = "Validation", prefix: str = "val", print_report: bool = False) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricsTracker()
        pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
        for batch in pbar:
            out = self._step_batch(batch, train=False)
            labels = batch["label"].to(cfg.device)
            preds = out["logits"].argmax(dim=-1)
            tracker.update(
                preds,
                labels,
                loss=float(out["loss"].item()),
                ce_loss=float(out["ce_loss"].item()),
                risk_loss=float(out["risk_loss"].item()),
            )
            m = tracker.compute(prefix)
            if m:
                pbar.set_postfix(
                    loss=f"{m[f'{prefix}_loss']:.4f}",
                    acc=f"{m[f'{prefix}_accuracy']:.4f}",
                    f1=f"{m[f'{prefix}_f1_macro']:.4f}",
                )

        metrics = tracker.compute(prefix)
        if print_report:
            print(f"\n[{desc}] Classification Report:")
            print(tracker.classification_report())
        return metrics

    def save_checkpoint(self, epoch: int, best_score: float, path: str, is_best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "class_weights": self.class_weights.detach().cpu() if self.class_weights is not None else None,
            "config": {
                "architecture": "FinReport-NextGen-v3",
                "n_classes": cfg.n_classes,
                "label_names": cfg.label_names,
                "label_mode": cfg.label_mode,
                "news_pooling": cfg.news_pooling,
                "max_news_per_day": cfg.max_news_per_day,
                "use_bidirectional_fusion": cfg.use_bidirectional_fusion,
            },
        }
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, "finreport_nextgen_v3_best.pt")

    def load_checkpoint(self, path: str = "checkpoint_v3.pt") -> tuple[int, float]:
        if not os.path.exists(path):
            print(f"[Checkpoint] no checkpoint found at {path}")
            return 1, 0.0
        ckpt = torch.load(path, map_location=cfg.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ckpt.get("class_weights") is not None:
            self.class_weights = ckpt["class_weights"].to(cfg.device)
            self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", 0.0))
        print(f"[Checkpoint] resumed from epoch {start_epoch - 1}, best_score={best_score:.4f}")
        return start_epoch, best_score

    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 20,
        resume: bool = True,
        checkpoint_path: str = "checkpoint_v3.pt",
        patience: int = 5,
        min_delta: float = 1e-4,
        es_metric: str = "val_f1_macro",
        es_mode: str = "max",
    ) -> None:
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode=es_mode, restore_best=True)
        start_epoch, best_score = (1, 0.0)
        if resume and os.path.exists(checkpoint_path):
            start_epoch, best_score = self.load_checkpoint(checkpoint_path)
        elif cfg.use_class_weights:
            self.set_class_weights_from_loader(train_loader)

        for epoch in range(start_epoch, n_epochs + 1):
            print(f"\n{'=' * 55}\n  EPOCH {epoch}/{n_epochs}\n{'=' * 55}")
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, desc=f"Epoch {epoch} [Validation]", prefix="val")

            print(
                f"\n[Epoch {epoch}] "
                f"Train Loss={train_metrics.get('train_loss', 0):.4f} "
                f"CE={train_metrics.get('train_ce_loss', 0):.4f} "
                f"Risk={train_metrics.get('train_risk_loss', 0):.4f} "
                f"Acc={train_metrics.get('train_accuracy', 0):.4f} "
                f"F1={train_metrics.get('train_f1_macro', 0):.4f}\n"
                f"          Val   Loss={val_metrics.get('val_loss', 0):.4f} "
                f"CE={val_metrics.get('val_ce_loss', 0):.4f} "
                f"Risk={val_metrics.get('val_risk_loss', 0):.4f} "
                f"Acc={val_metrics.get('val_accuracy', 0):.4f} "
                f"F1={val_metrics.get('val_f1_macro', 0):.4f} "
                f"Prec={val_metrics.get('val_precision_macro', 0):.4f} "
                f"Rec={val_metrics.get('val_recall_macro', 0):.4f}"
            )

            score = val_metrics.get(es_metric, val_metrics.get("val_f1_macro", 0.0))
            if best_score == 0.0 and es_mode == "min":
                is_best = True
            else:
                is_best = score > best_score if es_mode == "max" else score < best_score
            if is_best:
                best_score = score

            self.save_checkpoint(epoch=epoch, best_score=best_score, path=checkpoint_path, is_best=is_best)
            should_stop = early_stopping.step(score, self.model)
            if is_best:
                print(f"  ✓ New best model saved ({es_metric}={best_score:.4f})")
            if should_stop:
                print(f"[EarlyStopping] stopped at epoch {epoch}. Best {es_metric}={best_score:.4f}")
                break
