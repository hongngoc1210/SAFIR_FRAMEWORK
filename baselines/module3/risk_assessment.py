"""
Module 3 — Risk Assessment (EGARCH-VaR)
Adjusted for actual CSV column names:
  ff5_news_params : beta_NEWS, beta_MKT_RF, R2
  df_module2_news : ts_code_x, date_norm, excess_return, NEWS, ...
"""

import numpy as np
import pandas as pd
import warnings
from tqdm.auto import tqdm
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error, mean_absolute_error
from module3.egarch import EGARCHVaRModel

RF_ANNUAL = 0.0225
RF_DAILY  = RF_ANNUAL / 252

EGARCH_CONFIG = {
    'var': {
        'confidence_level': 0.05,
        'min_samples': 60,
        'max_zero_ratio': 0.1,
    },
    'egarch': {
        'p': 1,
        'q': 1,
        'dist': 't',
        'scale': 100,
        'rolling_window': 60,
    }
}

# ══════════════════════════════════════════════════════════════════
# BƯỚC 1: CHUẨN BỊ DATA
# ══════════════════════════════════════════════════════════════════

print("Preparing data from Module 2...")

df_module2_news = pd.read_csv("/kaggle/working/outputs/module2_data_ff5news.csv")
ff5_news_params = pd.read_csv("/kaggle/working/outputs/module2/ols_results_ff5news.csv")

df_m2 = df_module2_news.copy()
df_m2['date'] = pd.to_datetime(df_m2['date_norm'])
df_m2 = df_m2.sort_values(['ts_code_x', 'date'])

# Khôi phục raw return
df_m2['return'] = df_m2['excess_return'] + RF_DAILY

print(f"  Stocks: {df_m2['ts_code_x'].nunique()}")
print(f"  Date range: {df_m2['date'].min().date()} → {df_m2['date'].max().date()}")
print(f"  return stats: mean={df_m2['return'].mean():.6f}, std={df_m2['return'].std():.6f}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 2: EGARCH-VaR PER STOCK
# ══════════════════════════════════════════════════════════════════

print("\nRunning EGARCH-VaR per stock...")

var_model = EGARCHVaRModel(EGARCH_CONFIG)
var_results = []

for code, grp in tqdm(df_m2.groupby('ts_code_x'), desc="EGARCH-VaR"):
    grp = grp.sort_values('date')
    stock_df = grp[['date', 'return']].copy()

    try:
        predicted_var, actual_var = var_model.compute_var(stock_df)
        var_results.append({
            'ts_code_x'    : code,
            'n_obs'        : len(stock_df),
            'mean_return'  : float(stock_df['return'].mean()),
            'std_return'   : float(stock_df['return'].std()),
            'predicted_var': predicted_var,
            'actual_var'   : actual_var,
            'var_breach'   : int(actual_var < predicted_var),
        })
    except Exception as e:
        var_results.append({
            'ts_code_x'    : code,
            'n_obs'        : len(stock_df),
            'mean_return'  : float(stock_df['return'].mean()) if len(stock_df) > 0 else np.nan,
            'std_return'   : float(stock_df['return'].std()) if len(stock_df) > 0 else np.nan,
            'predicted_var': np.nan,
            'actual_var'   : np.nan,
            'var_breach'   : np.nan,
            'error'        : str(e),
        })

df_var = pd.DataFrame(var_results)
print(f"\nVaR computed: {df_var['predicted_var'].notna().sum()} / {len(df_var)} stocks")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 3: MERGE VỚI FF5-NEWS PARAMS
# ══════════════════════════════════════════════════════════════════

print("\nMerging with FF5-News factor loadings...")

# Kiểm tra cột thực tế
print("  ff5_news_params columns:", ff5_news_params.columns.tolist())

# Rename để chuẩn hoá (chữ hoa → chữ thường)
params_cols = {
    'ts_code_x' : 'ts_code_x',
    'alpha'     : 'alpha',
    'beta_MKT_RF': 'beta_mkt',   
    'beta_NEWS' : 'beta_news',   
    'R2'        : 'r2',          
}


available = {k: v for k, v in params_cols.items() if k in ff5_news_params.columns}
params_clean = ff5_news_params[list(available.keys())].rename(columns=available)

df_risk = df_var.merge(params_clean, on='ts_code_x', how='left')
print(f"  Merged columns: {df_risk.columns.tolist()}")
print(f"  beta_news notna: {df_risk['beta_news'].notna().sum()} / {len(df_risk)}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 4: KUPIEC POF BACKTEST
# ══════════════════════════════════════════════════════════════════

def kupiec_pof_test(returns, var_threshold, alpha=0.05):
    n = len(returns)
    n_breach = np.sum(returns < var_threshold)
    breach_rate = n_breach / n if n > 0 else np.nan

    if n_breach == 0 or n_breach == n:
        return n_breach, breach_rate, np.nan, False

    p     = alpha
    p_hat = breach_rate
    lr    = -2 * (
        n_breach * np.log(p / p_hat) +
        (n - n_breach) * np.log((1 - p) / (1 - p_hat))
    )
    pvalue = 1 - chi2.cdf(lr, df=1)
    reject = pvalue < 0.05

    return int(n_breach), float(breach_rate), float(pvalue), bool(reject)


print("\nRunning Kupiec backtest...")

backtest_results = []

for code, grp in tqdm(df_m2.groupby('ts_code_x'), desc="Kupiec backtest"):
    grp = grp.sort_values('date')

    row = df_risk[df_risk['ts_code_x'] == code]
    if row.empty or row['predicted_var'].isna().all():
        continue

    var_thresh = float(row['predicted_var'].iloc[0])
    returns    = grp['return'].values

    n_breach, breach_rate, pvalue, reject = kupiec_pof_test(
        returns, var_thresh, alpha=EGARCH_CONFIG['var']['confidence_level']
    )

    backtest_results.append({
        'ts_code_x'   : code,
        'n_obs'       : len(returns),
        'n_breach'    : n_breach,
        'breach_rate' : breach_rate,
        'kupiec_pval' : pvalue,
        'model_reject': reject,
    })

df_backtest = pd.DataFrame(backtest_results)

print(f"\nKupiec backtest summary:")
print(f"  Median breach rate : {df_backtest['breach_rate'].median():.1%}  (target: 5%)")
print(f"  Mean breach rate   : {df_backtest['breach_rate'].mean():.1%}")
print(f"  % model rejected   : {df_backtest['model_reject'].mean():.1%}")

breach_dist = pd.cut(
    df_backtest['breach_rate'],
    bins=[0, 0.03, 0.05, 0.08, 0.15, 1.0],
    labels=['<3% (conservative)', '3-5% (good)', '5-8% (acceptable)', '8-15% (loose)', '>15% (bad)']
).value_counts().sort_index()
print(f"\nBreach rate distribution:\n{breach_dist}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 4b: RISK CLASSIFICATION (ADAPTIVE PERCENTILE)
# ══════════════════════════════════════════════════════════════════

valid_var = df_risk['predicted_var'].dropna()
p33 = valid_var.quantile(0.33)
p67 = valid_var.quantile(0.67)

print(f"\nAdaptive VaR thresholds:")
print(f"  p33 = {p33:.4f}  (below → High risk)")
print(f"  p67 = {p67:.4f}  (above → Low risk)")

# News sensitivity từ beta_news percentile
bn      = df_risk['beta_news'].dropna()
bn_p33  = bn.quantile(0.33)
bn_p67  = bn.quantile(0.67)

def classify_news_sensitivity(b):
    if pd.isna(b): return 'Unknown'
    if b >= bn_p67: return 'High'
    elif b >= bn_p33: return 'Medium'
    else: return 'Low'

def classify_var_adaptive(var_val):
    if pd.isna(var_val): return 'Unknown'
    if var_val >= p67:   return 'Low'
    elif var_val >= p33: return 'Medium'
    else:                return 'High'

df_risk['news_sensitivity'] = df_risk['beta_news'].apply(classify_news_sensitivity)
df_risk['var_risk_level']   = df_risk['predicted_var'].apply(classify_var_adaptive)

print(f"\nNews sensitivity:\n{df_risk['news_sensitivity'].value_counts()}")
print(f"\nRisk level:\n{df_risk['var_risk_level'].value_counts()}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 5: MERGE BACKTEST VÀO RISK REPORT
# ══════════════════════════════════════════════════════════════════

df_risk = df_risk.merge(
    df_backtest[['ts_code_x', 'n_breach', 'breach_rate', 'kupiec_pval', 'model_reject']],
    on='ts_code_x',
    how='left'
)

print(f"\nPoorly calibrated VaR models: "
      f"{df_risk['model_reject'].sum()} / {df_risk['model_reject'].notna().sum()}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 6: FINAL RISK SUMMARY
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODULE 3 — FINAL RISK REPORT")
print("="*60)

port_var = df_risk['predicted_var'].mean()
print(f"\nPortfolio-level VaR (equal-weighted, daily 5%):")
print(f"  Mean VaR : {port_var:.4f} ({port_var*100:.2f}%/day)")
print(f"  Annualized (×√252) : {port_var * np.sqrt(252):.4f}")

print(f"\nRisk segmentation:")
seg = df_risk.groupby('var_risk_level').agg(
    n_stocks       = ('ts_code_x',     'count'),
    mean_var       = ('predicted_var',  'mean'),
    mean_beta_news = ('beta_news',      'mean'),
    mean_breach    = ('breach_rate',    'mean'),
).round(4)
print(seg)

print(f"\nRisk × News Sensitivity cross-tab:")
ct = pd.crosstab(df_risk['var_risk_level'], df_risk['news_sensitivity'])
print(ct)

# ══════════════════════════════════════════════════════════════════
# BƯỚC 7: SAVE 
# ══════════════════════════════════════════════════════════════════

import os
out_dir = '/kaggle/working/outputs'
os.makedirs(out_dir, exist_ok=True)

df_risk.to_csv(os.path.join(out_dir, 'module3_var_risk_report.csv'), index=False)
df_backtest.to_csv(os.path.join(out_dir, 'module3_kupiec_backtest.csv'), index=False)

print(f"\n✓ Saved:")
print(f"  module3_var_risk_report.csv  ({len(df_risk)} stocks)")
print(f"  module3_kupiec_backtest.csv  ({len(df_backtest)} stocks)")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 8: EVALUATION METRICS 
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("VaR EVALUATION METRICS")
print("="*60)

df_eval = df_risk.dropna(subset=['predicted_var', 'actual_var']).copy()
N = len(df_eval)
print(f"\nStocks dùng để evaluate: {N} / {len(df_risk)}")

rmse_dec = np.sqrt(mean_squared_error(df_eval['actual_var'], df_eval['predicted_var']))
mae_dec  = mean_absolute_error(df_eval['actual_var'], df_eval['predicted_var'])
rmse_pct = rmse_dec * 100
mae_pct  = mae_dec  * 100

var_loss_dec = (df_eval['predicted_var'] - df_eval['actual_var']).abs().mean()
var_loss_pct = var_loss_dec * 100

coverage_stock = (df_eval['predicted_var'] <= df_eval['actual_var']).mean()
coverage_day   = 1 - df_backtest['breach_rate'].mean()

print(f"""
┌─────────────────────────┬──────────────┬───────────────┐
│ Metric                  │  Code (dec)  │  Code (×100)  │
├─────────────────────────┼──────────────┼───────────────┤
│ RMSE                    │   {rmse_dec:.4f}     │    {rmse_pct:.4f}     │
│ MAE                     │   {mae_dec:.4f}     │    {mae_pct:.4f}     │
│ VaR Loss (Eq.12)        │   {var_loss_dec:.4f}     │    {var_loss_pct:.4f}     │
│ Coverage Rate (stock)   │   {coverage_stock:.4f}     │      —        │
│ Coverage Rate (day-lvl) │   {coverage_day:.4f}     │      —        │
└─────────────────────────┴──────────────┴───────────────┘
""")

bias     = (df_eval['predicted_var'] - df_eval['actual_var']).mean()
bias_pct = bias * 100
print(f"Bias analysis:")
print(f"  Mean bias (predicted − actual) : {bias:.6f} ({bias_pct:+.4f}%)")
print(f"  → {'Model OVER-estimates loss (conservative ✅)' if bias < 0 else 'Model UNDER-estimates loss (risky ⚠️)'}")

df_eval['covered'] = df_eval['predicted_var'] <= df_eval['actual_var']
cov_by_level = df_eval.groupby('var_risk_level')['covered'].agg(['sum','count','mean'])
cov_by_level.columns = ['n_covered', 'n_total', 'coverage_rate']
cov_by_level['coverage_rate'] = cov_by_level['coverage_rate'].map('{:.1%}'.format)
print(f"\nCoverage rate theo Risk Level:\n{cov_by_level.to_string()}")

if 'news_sensitivity' in df_eval.columns:
    cov_by_news = df_eval.groupby('news_sensitivity')['covered'].agg(['sum','count','mean'])
    cov_by_news.columns = ['n_covered', 'n_total', 'coverage_rate']
    cov_by_news['coverage_rate'] = cov_by_news['coverage_rate'].map('{:.1%}'.format)
    print(f"\nCoverage rate theo News Sensitivity:\n{cov_by_news.to_string()}")

eval_summary = {
    'N_stocks'           : N,
    'RMSE_decimal'       : round(rmse_dec, 6),
    'RMSE_pct'           : round(rmse_pct, 4),
    'MAE_decimal'        : round(mae_dec, 6),
    'MAE_pct'            : round(mae_pct, 4),
    'VaR_Loss_decimal'   : round(var_loss_dec, 6),
    'VaR_Loss_pct'       : round(var_loss_pct, 4),
    'Coverage_Rate_stock': round(float(coverage_stock), 4),
    'Coverage_Rate_day'  : round(float(coverage_day), 4),
    'Bias_decimal'       : round(float(bias), 6),
    'Paper_RMSE'         : 0.0947,
    'Paper_MAE'          : 0.8176,
    'Paper_Coverage'     : 0.8123,
}

pd.DataFrame([eval_summary]).to_csv(
    '/kaggle/working/outputs/module3_eval_metrics.csv', index=False
)
print("\n✓ Saved: module3_eval_metrics.csv")
