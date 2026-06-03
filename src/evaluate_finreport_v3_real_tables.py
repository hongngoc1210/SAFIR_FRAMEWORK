from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# ---------------------------------------------------------------------------
# Robust package imports
# ---------------------------------------------------------------------------

def _import_package_modules():
    """Import project modules whether this file is run as module or script."""
    if __package__:
        base = __package__
    else:
        script_dir = Path(__file__).resolve().parent
        parent = script_dir.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        base = script_dir.name

    try:
        cfg_mod = importlib.import_module(f"{base}.config")
        data_mod = importlib.import_module(f"{base}.data_preprocessing")
        model_mod = importlib.import_module(f"{base}.model_main")
    except Exception:
        # Fallback for flat project layout: config.py/model_main.py in cwd.
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        cfg_mod = importlib.import_module("config")
        data_mod = importlib.import_module("data_preprocessing")
        model_mod = importlib.import_module("model_main")

    return cfg_mod.Config, data_mod.run_preprocessing, model_mod.FinReportNextGen


cfg, run_preprocessing, FinReportNextGen = _import_package_modules()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper-style real evaluation for FinReport-NextGen v3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", type=str, required=True, help="Folder containing train.csv/val.csv/test.csv")
    p.add_argument("--full-model", type=str, required=True, help="Full ISF+DMAQ checkpoint")
    p.add_argument("--factors-model", type=str, default=None, help="Optional factors-only DMAQ checkpoint")
    p.add_argument("--news-model", type=str, default=None, help="Optional news-only ISF checkpoint")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--output-dir", type=str, default="paper_style_real_eval")

    p.add_argument("--label-mode", type=str, default="binary", choices=["binary", "ternary"])
    p.add_argument("--news-pooling", type=str, default="sap", choices=["sap", "cap", "pa_sap"])
    p.add_argument("--max-news-per-day", type=int, default=4)
    p.add_argument("--max-text-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])

    # Return construction.
    p.add_argument("--return-col", type=str, default=None, help="Use this column from raw split as realized return if present")
    p.add_argument("--risk-free-col", type=str, default=None, help="Optional risk-free return column")
    p.add_argument("--return-from-price", action="store_true", help="Force realized return from price window instead of raw return column")

    # Factor construction / GRS.
    p.add_argument("--score-quantile", type=float, default=0.2, help="Top/bottom quantile for model-implied factor")
    p.add_argument("--min-stock-obs", type=int, default=10, help="Minimum observations per stock for regression/GRS")
    p.add_argument("--max-grs-assets", type=int, default=50, help="Maximum stock assets for GRS; avoids T <= N+K issues")
    p.add_argument("--portfolio-assets", action="store_true", help="Use score-quantile portfolios as GRS test assets instead of individual stocks")
    p.add_argument("--n-portfolios", type=int, default=5, help="Number of daily score portfolios if --portfolio-assets")

    # Backtest.
    p.add_argument("--top-k", type=int, default=None, help="Daily top-k stocks to buy; if absent, use --long-quantile")
    p.add_argument("--long-quantile", type=float, default=0.2, help="Daily top quantile to buy")
    p.add_argument("--transaction-cost", type=float, default=0.001, help="Daily transaction cost subtracted from portfolio return")
    p.add_argument("--annualization", type=int, default=252)

    p.add_argument("--max-batches", type=int, default=None, help="Debug: evaluate only first N batches")
    p.add_argument("--save-predictions", action="store_true", help="Save per-sample prediction CSV")
    return p.parse_args()


def apply_cli_config(args: argparse.Namespace) -> None:
    if args.device is not None:
        cfg.device = args.device
    elif getattr(cfg, "device", "cuda") == "cuda" and not torch.cuda.is_available():
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


# ---------------------------------------------------------------------------
# Model/data collection
# ---------------------------------------------------------------------------

def select_loader(loaders: Tuple[Any, Any, Any], split: str):
    return {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]


def load_model(checkpoint_path: str, n_codes: int) -> torch.nn.Module:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    print(f"[Model] Loading: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=cfg.device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = FinReportNextGen(n_codes=n_codes, enable_sefn=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Model] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Model] Unexpected keys: {len(unexpected)}")
    model.to(cfg.device)
    model.eval()
    return model


def realized_return_from_sample_price(sample: Dict[str, Any]) -> float:
    """Fallback return from the raw price window stored in dataset samples."""
    arr = np.asarray(sample.get("price_seq"), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return 0.0
    valid = arr[:, 0] > 1e-8
    if not valid.any():
        return 0.0
    last = arr[valid][-1]
    open_, close = float(last[0]), float(last[1])
    if abs(open_) < 1e-12:
        return 0.0
    return float((close - open_) / open_)


def batch_to_device(batch: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(cfg.device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def collect_model_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    args: argparse.Namespace,
    zero_news: bool = False,
    zero_price: bool = False,
    zero_market: bool = False,
) -> pd.DataFrame:
    """Run one checkpoint and collect per-sample scores/returns/risk outputs."""
    rows: List[Dict[str, Any]] = []
    dataset = loader.dataset
    samples = getattr(dataset, "samples", None)
    sample_offset = 0

    pbar = tqdm(loader, desc=f"Predict {model_name}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        b = batch_to_device(batch)

        price_seq = b["price_seq"]
        mkt_vector = b["mkt_vector"]
        if zero_price:
            price_seq = torch.zeros_like(price_seq)
        if zero_market:
            mkt_vector = torch.zeros_like(mkt_vector)

        if zero_news:
            news_factor = torch.zeros((price_seq.size(0), cfg.news_factor_dim), device=cfg.device)
            dmaq_out = model.dmaq(price_seq, mkt_vector, news_factor, code_ids=b.get("code_id"))
            logits = dmaq_out["logits"]
            out = {
                "logits": logits,
                "pred_labels": logits.argmax(dim=-1),
                "confidence": logits.softmax(dim=-1).max(dim=-1).values,
                "news_factor": news_factor,
                "quant_factors": dmaq_out["quant_factors"],
                "log_vol": dmaq_out["log_vol"],
                "var_est": dmaq_out["var_est"],
            }
        else:
            out = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                code_ids=b["code_id"],
                price_seq=price_seq,
                mkt_vector=mkt_vector,
                news_item_mask=b.get("news_item_mask"),
                generate_text=False,
            )

        logits = out["logits"].float()
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values
        if cfg.n_classes == 2:
            score = probs[:, 1] - probs[:, 0]
        else:
            # positive minus negative, neutral ignored for ranking score
            score = probs[:, -1] - probs[:, 0]

        B = logits.size(0)
        codes = batch.get("code", [None] * B)
        dates = batch.get("date", [None] * B)
        if isinstance(codes, tuple):
            codes = list(codes)
        if isinstance(dates, tuple):
            dates = list(dates)

        labels = b["label"].detach().cpu().numpy()
        var_est = out["var_est"].detach().cpu().numpy()
        log_vol = out["log_vol"].detach().cpu().numpy()
        var_target = b.get("var_target")
        var_target_np = var_target.detach().cpu().numpy() if isinstance(var_target, torch.Tensor) else np.full(B, np.nan)

        for i in range(B):
            # For val/test loaders shuffle=False, this aligns with dataset.samples.
            sample = samples[sample_offset + i] if samples is not None and sample_offset + i < len(samples) else {}
            realized_ret = realized_return_from_sample_price(sample)
            rows.append(
                {
                    "model": model_name,
                    "code": str(codes[i]) if codes is not None else str(sample.get("CODE", "")),
                    "date": str(dates[i])[:10] if dates is not None else str(sample.get("trade_date", ""))[:10],
                    "label": int(labels[i]),
                    "pred": int(pred[i].detach().cpu().item()),
                    "score": float(score[i].detach().cpu().item()),
                    "confidence": float(conf[i].detach().cpu().item()),
                    "prob_up": float(probs[i, min(1, probs.size(1)-1)].detach().cpu().item()),
                    "realized_return": float(realized_ret),
                    "var_est_loss": float(var_est[i]),
                    "var_threshold": float(-abs(var_est[i])),
                    "log_vol_pred": float(log_vol[i]),
                    "var_target_loss": float(var_target_np[i]),
                }
            )
        sample_offset += B

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Return factor construction and GRS
# ---------------------------------------------------------------------------

def make_model_factor_return(df: pd.DataFrame, q: float = 0.2) -> pd.Series:
    """Daily long-short factor: top-score return minus bottom-score return."""
    values = []
    for date, g in df.groupby("date"):
        g = g.dropna(subset=["score", "realized_return"])
        if len(g) < 4:
            continue
        n = max(1, int(math.floor(len(g) * q)))
        gs = g.sort_values("score")
        bottom = gs.head(n)["realized_return"].mean()
        top = gs.tail(n)["realized_return"].mean()
        values.append((date, float(top - bottom)))
    return pd.Series(dict(values), name="model_factor_return").sort_index()


def make_market_return(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["realized_return"].mean().sort_index().rename("market_return")


def ols_regression(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return beta including intercept, residuals, R^2."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    yhat = X1 @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return beta, resid, r2


def build_grs_panel_individual_stocks(
    df: pd.DataFrame,
    factors: pd.DataFrame,
    min_obs: int,
    max_assets: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build balanced stock-return panel and aligned factor matrix."""
    pivot = df.pivot_table(index="date", columns="code", values="realized_return", aggfunc="mean").sort_index()
    counts = pivot.notna().sum().sort_values(ascending=False)
    cols = counts[counts >= min_obs].index.tolist()[:max_assets]
    R = pivot[cols].copy()
    joined = R.join(factors, how="inner")
    joined = joined.dropna(axis=0, how="any")
    R_bal = joined[cols]
    X = joined[factors.columns]
    return R_bal, X


def build_grs_panel_score_portfolios(
    df: pd.DataFrame,
    factors: pd.DataFrame,
    n_portfolios: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily score-sorted test assets as portfolios."""
    records = []
    for date, g in df.groupby("date"):
        g = g.dropna(subset=["score", "realized_return"])
        if len(g) < n_portfolios:
            continue
        # qcut may fail with duplicated scores; rank first.
        ranks = g["score"].rank(method="first")
        bins = pd.qcut(ranks, q=n_portfolios, labels=False, duplicates="drop")
        tmp = g.copy()
        tmp["portfolio"] = bins
        for p, gg in tmp.groupby("portfolio"):
            records.append({"date": date, f"P{int(p)+1}": gg["realized_return"].mean()})
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    port = pd.DataFrame(records).groupby("date").mean().sort_index()
    joined = port.join(factors, how="inner").dropna(axis=0, how="any")
    R = joined[port.columns]
    X = joined[factors.columns]
    return R, X


def compute_explanatory_power(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, float]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    model_factor = make_model_factor_return(df, q=args.score_quantile)
    market = make_market_return(df)
    factors = pd.concat([market, model_factor], axis=1).dropna()

    if args.portfolio_assets:
        R, X = build_grs_panel_score_portfolios(df, factors, n_portfolios=args.n_portfolios)
    else:
        R, X = build_grs_panel_individual_stocks(
            df, factors, min_obs=args.min_stock_obs, max_assets=args.max_grs_assets
        )
        # Fallback to portfolios if individual-stock panel is too small.
        if R.empty or len(R) <= R.shape[1] + X.shape[1] + 1:
            R, X = build_grs_panel_score_portfolios(df, factors, n_portfolios=args.n_portfolios)

    if R.empty or X.empty:
        return {
            "GRS": float("nan"),
            "GRS p-value": float("nan"),
            "Mean Absolute Value of Alpha": float("nan"),
            "Mean R^2": float("nan"),
            "T": 0,
            "N assets": 0,
            "K factors": 0,
        }

    y_mat = R.to_numpy(dtype=float)
    x_mat = X.to_numpy(dtype=float)
    T, N = y_mat.shape
    K = x_mat.shape[1]

    alphas = []
    residuals = []
    r2s = []
    for j in range(N):
        beta, resid, r2 = ols_regression(y_mat[:, j], x_mat)
        alphas.append(beta[0])
        residuals.append(resid)
        r2s.append(r2)

    alpha = np.asarray(alphas).reshape(-1, 1)
    E = np.column_stack(residuals)
    sigma = (E.T @ E) / max(T - K - 1, 1)
    f_mean = x_mat.mean(axis=0).reshape(-1, 1)
    omega = np.cov(x_mat, rowvar=False)
    if K == 1:
        omega = np.asarray([[float(omega)]])

    try:
        sigma_inv = np.linalg.pinv(sigma)
        omega_inv = np.linalg.pinv(omega)
        numerator = float(alpha.T @ sigma_inv @ alpha)
        denominator = float(1.0 + f_mean.T @ omega_inv @ f_mean)
        grs = float(((T - N - K) / max(N, 1)) * numerator / max(denominator, 1e-12))
        if stats is not None and T - N - K > 0:
            pval = float(1.0 - stats.f.cdf(grs, N, T - N - K))
        else:
            pval = float("nan")
    except Exception:
        grs, pval = float("nan"), float("nan")

    return {
        "GRS": grs,
        "GRS p-value": pval,
        "Mean Absolute Value of Alpha": float(np.mean(np.abs(alphas))),
        "Mean R^2": float(np.mean(r2s)),
        "T": int(T),
        "N assets": int(N),
        "K factors": int(K),
    }


# ---------------------------------------------------------------------------
# VaR and backtest
# ---------------------------------------------------------------------------

def kupiec_pof_test(n_breaches: int, n_obs: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    """Kupiec unconditional coverage likelihood ratio test."""
    if n_obs <= 0:
        return float("nan"), float("nan"), False
    x = int(n_breaches)
    n = int(n_obs)
    phat = x / n
    if phat <= 0 or phat >= 1:
        # Handle boundary with small epsilon.
        phat = min(max(phat, 1e-12), 1 - 1e-12)
    try:
        logL0 = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
        logL1 = (n - x) * math.log(1 - phat) + x * math.log(phat)
        lr = -2.0 * (logL0 - logL1)
        p = float(1.0 - stats.chi2.cdf(lr, 1)) if stats is not None else float("nan")
        return float(lr), p, bool(p < 0.05) if not math.isnan(p) else False
    except Exception:
        return float("nan"), float("nan"), False


def compute_var_tables(df: pd.DataFrame, annualization: int = 252, alpha: float = 0.05) -> Tuple[Dict[str, float], Dict[str, float]]:
    out = df.copy()
    out["actual_loss"] = (-out["realized_return"]).clip(lower=0.0)
    out["var_pred_loss"] = out["var_est_loss"].abs()
    target = out["var_target_loss"].where(out["var_target_loss"].notna(), out["actual_loss"])
    err = out["var_pred_loss"] - target

    breach = out["realized_return"] < -out["var_pred_loss"]
    coverage = 1.0 - float(breach.mean())
    var_loss = np.maximum(0.0, out["actual_loss"].to_numpy() - out["var_pred_loss"].to_numpy()).mean()

    per_stock = []
    for code, g in out.groupby("code"):
        b = (g["realized_return"] < -g["var_pred_loss"]).sum()
        n = len(g)
        _, p, reject = kupiec_pof_test(int(b), int(n), alpha=alpha)
        per_stock.append({"code": code, "breach_rate": b / n if n else np.nan, "reject": reject, "p": p})
    ps = pd.DataFrame(per_stock)

    metrics = {
        "RMSE": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
        "MAE": float(np.mean(np.abs(err.to_numpy()))),
        "VaR Loss": float(var_loss),
        "Coverage Rate": float(coverage),
    }
    summary = {
        "Stocks evaluated": int(out["code"].nunique()),
        "Mean VaR (5%)": float(-out["var_pred_loss"].mean()),
        "Annualized VaR": float(-out["var_pred_loss"].mean() * math.sqrt(annualization)),
        "Mean breach rate": float(ps["breach_rate"].mean()) if not ps.empty else float("nan"),
        "Median breach rate": float(ps["breach_rate"].median()) if not ps.empty else float("nan"),
        "Rejected by Kupiec test": float(ps["reject"].mean()) if not ps.empty else float("nan"),
        "Coverage rate (day-level)": float(coverage),
        "Coverage rate (stock-level)": float(1.0 - ps["breach_rate"].mean()) if not ps.empty else float("nan"),
    }
    return metrics, summary


def max_drawdown(cum: pd.Series) -> float:
    if cum.empty:
        return float("nan")
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return float(dd.min())


def compute_backtest(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[Dict[str, float], pd.DataFrame]:
    daily = []
    for date, g in df.groupby("date"):
        g = g.dropna(subset=["score", "realized_return"]).sort_values("score", ascending=False)
        if g.empty:
            continue
        if args.top_k is not None:
            selected = g.head(args.top_k)
        else:
            n = max(1, int(math.ceil(len(g) * args.long_quantile)))
            selected = g.head(n)
        raw_ret = float(selected["realized_return"].mean())
        net_ret = raw_ret - float(args.transaction_cost)
        daily.append(
            {
                "date": date,
                "n_selected": int(len(selected)),
                "portfolio_return": net_ret,
                "raw_portfolio_return": raw_ret,
                "avg_score": float(selected["score"].mean()),
            }
        )
    d = pd.DataFrame(daily).sort_values("date")
    if d.empty:
        return {
            "Maximum Drawdown": float("nan"),
            "Annualized Rate of Return": float("nan"),
            "Sharpe Ratio": float("nan"),
            "Cumulative Return": float("nan"),
            "Win Rate": float("nan"),
        }, d
    d["cumulative_return"] = (1.0 + d["portfolio_return"]).cumprod()
    n = len(d)
    cumulative = float(d["cumulative_return"].iloc[-1] - 1.0)
    annualized = float(d["cumulative_return"].iloc[-1] ** (args.annualization / max(n, 1)) - 1.0)
    std = float(d["portfolio_return"].std(ddof=1))
    sharpe = float(math.sqrt(args.annualization) * d["portfolio_return"].mean() / std) if std > 1e-12 else float("nan")
    metrics = {
        "Maximum Drawdown": max_drawdown(d["cumulative_return"]),
        "Annualized Rate of Return": annualized,
        "Sharpe Ratio": sharpe,
        "Cumulative Return": cumulative,
        "Win Rate": float((d["portfolio_return"] > 0).mean()),
        "Trading Days": int(n),
    }
    return metrics, d


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x * 100:.{digits}f}%"


def save_tables(
    return_table: pd.DataFrame,
    risk_table: pd.DataFrame,
    var_summary: pd.DataFrame,
    backtest_table: pd.DataFrame,
    daily_backtests: Dict[str, pd.DataFrame],
    predictions: Dict[str, pd.DataFrame],
    notes: Dict[str, Any],
    output_dir: Path,
    save_predictions: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = output_dir / "finreport_v3_paper_style_real_eval.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        return_table.to_excel(writer, sheet_name="Return_Forecasting", index=False)
        risk_table.to_excel(writer, sheet_name="VaR_Risk_Assessment", index=False)
        var_summary.to_excel(writer, sheet_name="VaR_Summary", index=False)
        backtest_table.to_excel(writer, sheet_name="Backtest", index=False)
        for name, daily in daily_backtests.items():
            safe = name[:25].replace("/", "_").replace("\\", "_")
            daily.to_excel(writer, sheet_name=f"Daily_{safe}", index=False)

    return_table.to_csv(output_dir / "return_forecasting_grs_table.csv", index=False, encoding="utf-8-sig")
    risk_table.to_csv(output_dir / "var_risk_assessment_table.csv", index=False, encoding="utf-8-sig")
    var_summary.to_csv(output_dir / "var_summary_table.csv", index=False, encoding="utf-8-sig")
    backtest_table.to_csv(output_dir / "backtest_table.csv", index=False, encoding="utf-8-sig")

    if save_predictions:
        for name, df in predictions.items():
            df.to_csv(output_dir / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "evaluation_notes.json").open("w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

    md_path = output_dir / "finreport_v3_paper_style_real_eval.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# FinReport-NextGen v3 Paper-style Real Evaluation\n\n")
        f.write("## Return Forecasting / Explanatory Power\n\n")
        f.write(return_table.to_markdown(index=False))
        f.write("\n\n## VaR Risk Assessment\n\n")
        f.write(risk_table.to_markdown(index=False))
        f.write("\n\n## VaR Summary\n\n")
        f.write(var_summary.to_markdown(index=False))
        f.write("\n\n## Backtest in Real-world Scenario Style\n\n")
        f.write(backtest_table.to_markdown(index=False))
        f.write("\n")

    print(f"[Saved] Excel: {xlsx_path}")
    print(f"[Saved] Markdown: {md_path}")


def print_tables(return_table: pd.DataFrame, risk_table: pd.DataFrame, var_summary: pd.DataFrame, backtest_table: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("RETURN FORECASTING / EXPLANATORY POWER")
    print("=" * 80)
    print(return_table.to_markdown(index=False))
    print("\n" + "=" * 80)
    print("VAR RISK ASSESSMENT")
    print("=" * 80)
    print(risk_table.to_markdown(index=False))
    print("\n" + "=" * 80)
    print("VAR SUMMARY")
    print("=" * 80)
    print(var_summary.to_markdown(index=False))
    print("\n" + "=" * 80)
    print("BACKTEST IN REAL-WORLD SCENARIO STYLE")
    print("=" * 80)
    print(backtest_table.to_markdown(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_cli_config(args)

    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print("=" * 80)
    print("FinReport-NextGen v3 | Paper-style Real Evaluation")
    print("=" * 80)
    print(f"Data path         : {data_path}")
    print(f"Split             : {args.split}")
    print(f"Full model        : {args.full_model}")
    print(f"Factors model     : {args.factors_model}")
    print(f"News model        : {args.news_model}")
    print(f"Device            : {cfg.device}")
    print(f"Label mode        : {cfg.label_mode} {cfg.label_names}")
    print(f"News pooling      : {cfg.news_pooling}")
    print("=" * 80)

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

    model_specs: List[Tuple[str, str, bool, bool, bool]] = []
    if args.factors_model:
        model_specs.append(("Ours-Factors", args.factors_model, True, False, False))
    if args.news_model:
        # For a news-only checkpoint, keep normal input. It should have been trained as news-only.
        model_specs.append(("Ours-News", args.news_model, False, False, False))
    model_specs.append(("Ours-News+Factors", args.full_model, False, False, False))

    predictions: Dict[str, pd.DataFrame] = {}
    for name, ckpt, zero_news, zero_price, zero_market in model_specs:
        model = load_model(ckpt, n_codes=n_codes)
        df_pred = collect_model_predictions(
            model=model,
            loader=loader,
            model_name=name,
            args=args,
            zero_news=False,  # checkpoint itself defines the trained module setting
            zero_price=False,
            zero_market=False,
        )
        predictions[name] = df_pred
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Return forecasting table: one row per model checkpoint.
    ret_rows = []
    for name, df in predictions.items():
        res = compute_explanatory_power(df, args)
        ret_rows.append(
            {
                "Model": name,
                "GRS": res["GRS"],
                "GRS p-value": res["GRS p-value"],
                "Mean Absolute Value of Alpha": res["Mean Absolute Value of Alpha"],
                "Mean R²": res["Mean R^2"],
                "T": res["T"],
                "N assets": res["N assets"],
            }
        )
    return_table = pd.DataFrame(ret_rows)

    # Risk and backtest: usually from full model, but include all supplied models for comparison.
    risk_rows = []
    var_summary_rows = []
    backtest_rows = []
    daily_backtests: Dict[str, pd.DataFrame] = {}
    for name, df in predictions.items():
        risk, varsum = compute_var_tables(df, annualization=args.annualization, alpha=0.05)
        risk_rows.append({"Model": name, **risk})
        var_summary_rows.extend([{"Model": name, "Metric": k, "Value": v} for k, v in varsum.items()])

        bt, daily = compute_backtest(df, args)
        backtest_rows.append({"Model": name, **bt})
        daily_backtests[name] = daily

    risk_table = pd.DataFrame(risk_rows)
    var_summary = pd.DataFrame(var_summary_rows)
    backtest_table = pd.DataFrame(backtest_rows)

    notes = {
        "important_note": (
            "This script does not fabricate Fama-French 5 factors. It evaluates the new architecture's "
            "own model-implied factors using paper-style GRS/alpha/R2, VaR, and backtest metrics. "
            "For strict FF5-News evaluation, provide true FF5 factor returns and implement the exact "
            "Fama-French portfolio construction from the target market."
        ),
        "factor_return_definition": "Daily top-score quantile return minus bottom-score quantile return.",
        "backtest_definition": "Daily long-only top-score portfolio, equal weighted, transaction cost subtracted daily.",
        "score_quantile": args.score_quantile,
        "long_quantile": args.long_quantile,
        "top_k": args.top_k,
        "transaction_cost": args.transaction_cost,
        "split": args.split,
    }

    save_tables(
        return_table=return_table,
        risk_table=risk_table,
        var_summary=var_summary,
        backtest_table=backtest_table,
        daily_backtests=daily_backtests,
        predictions=predictions,
        notes=notes,
        output_dir=output_dir,
        save_predictions=args.save_predictions,
    )
    print_tables(return_table, risk_table, var_summary, backtest_table)


if __name__ == "__main__":
    main()
