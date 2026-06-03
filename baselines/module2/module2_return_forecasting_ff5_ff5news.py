"""
Module 2: Return Forecasting — FF5 vs FF5-News Regression
===========================================================
Theo FinReport paper (WWW 2024) — Section 3.3 & Table 2.
"""

import numpy as np
import pandas as pd
import os
import glob
import warnings
from scipy import stats
import statsmodels.api as sm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

IN_DIR  = r'/kaggle/working/outputs'
OUT_DIR = r'/kaggle/working/outputs/module2'
os.makedirs(OUT_DIR, exist_ok=True)

RF_ANNUAL  = 0.0225
RF_MONTHLY = (1 + RF_ANNUAL) ** (1/12) - 1

FF5_COLS  = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
FF5N_COLS = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'NEWS']

MIN_OBS = 12


# ══════════════════════════════════════════════════════════════════
# BƯỚC 0: LOAD DATA
# ══════════════════════════════════════════════════════════════════

df_ff5  = pd.read_csv(os.path.join(IN_DIR, 'module2_data_rebuilt.csv'))
df_ff5n = pd.read_csv(os.path.join(IN_DIR, 'module2_data_ff5news.csv'))

df_ff5['date']  = pd.to_datetime(df_ff5['date_norm'])
df_ff5n['date'] = pd.to_datetime(df_ff5n['date_norm'])
df_ff5['ym']    = df_ff5['date'].dt.to_period('M')
df_ff5n['ym']   = df_ff5n['date'].dt.to_period('M')


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1: AGGREGATE DAILY → MONTHLY
# ══════════════════════════════════════════════════════════════════

def compound_ret(s):
    return (1 + s).prod() - 1

def aggregate_monthly(df, factor_cols):
    ret_m = (
        df.groupby(['ts_code_x', 'ym'])['excess_return']
        .apply(compound_ret)
        .reset_index()
        .rename(columns={'excess_return': 'exc_ret_m'})
    )
    fac_m = (
        df.drop_duplicates(subset=['date'])
        .sort_values('date')
        [['date', 'ym'] + factor_cols]
        .dropna(subset=factor_cols)
        .groupby('ym')[factor_cols]
        .apply(compound_ret)
        .reset_index()
    )
    return ret_m.merge(fac_m, on='ym', how='inner')

monthly_ff5  = aggregate_monthly(df_ff5,  FF5_COLS)
monthly_ff5n = aggregate_monthly(df_ff5n, FF5N_COLS)

T_ff5  = monthly_ff5['ym'].nunique()
T_ff5n = monthly_ff5n['ym'].nunique()


# ══════════════════════════════════════════════════════════════════
# BƯỚC 2: XÂY TEST PORTFOLIOS (5×5 Size × B/M)
# ══════════════════════════════════════════════════════════════════

def build_test_portfolios(df_daily, factor_cols, T_months):
    K     = len(factor_cols)
    N_max = T_months - K - 2

    raw_paths = [
        r'/kaggle/working/stock_daily/stock_daily1',
        r'/kaggle/working/stock_daily/stock_daily2',
        r'/kaggle/working/stock_daily/stock_daily3',
    ]
    dfs_raw = []
    for p in raw_paths:
        if not os.path.isdir(p):
            continue
        for f in sorted(glob.glob(os.path.join(p, '*.csv')))[:150]:
            try:
                dfs_raw.append(pd.read_csv(
                    f, sep='\t', encoding='utf-8',
                    usecols=lambda c: c in ['ts_code_x','trade_date','total_mv','pb','pct_chg']
                ))
            except:
                pass
    if not dfs_raw:
        return None, None

    df_raw = pd.concat(dfs_raw, ignore_index=True)
    df_raw['date'] = pd.to_datetime(df_raw['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    df_raw['ym']   = df_raw['date'].dt.to_period('M')
    df_raw['bm']   = np.where(df_raw['pb'] > 0, 1.0 / df_raw['pb'], np.nan)
    df_raw['ret']  = df_raw['pct_chg'] / 100.0
    df_raw = df_raw.dropna(subset=['total_mv', 'bm', 'ret'])
    df_raw = df_raw[df_raw['total_mv'] > 0]

    date_min = df_daily['date'].min()
    date_max = df_daily['date'].max()
    df_raw = df_raw[(df_raw['date'] >= date_min) & (df_raw['date'] <= date_max)]

    eom = (
        df_raw.sort_values('date')
        .groupby(['ts_code_x', 'ym']).last()
        .reset_index()[['ts_code_x', 'ym', 'total_mv', 'bm']]
        .dropna()
    )
    eom['ym_next'] = eom['ym'] + 1

    def quintile_cut(series, labels):
        try:
            return pd.qcut(series, q=5, labels=labels, duplicates='drop')
        except:
            return pd.Series(labels[2], index=series.index)

    size_labels = ['S1','S2','S3','S4','S5']
    bm_labels   = ['B1','B2','B3','B4','B5']

    eom['size_q'] = eom.groupby('ym')['total_mv'].transform(
        lambda x: quintile_cut(x, size_labels))
    eom['bm_q']   = eom.groupby('ym')['bm'].transform(
        lambda x: quintile_cut(x, bm_labels))

    sort_df = eom[['ts_code_x','ym_next','size_q','bm_q']].rename(columns={'ym_next':'ym'})
    df_m    = df_raw.merge(sort_df, on=['ts_code_x','ym'], how='left')
    df_m    = df_m.dropna(subset=['size_q','bm_q'])

    def vw(grp):
        mv  = grp['total_mv'].values
        ret = grp['ret'].values
        valid = np.isfinite(mv) & np.isfinite(ret) & (mv > 0)
        if valid.sum() == 0:
            return np.nan
        w = mv[valid] / mv[valid].sum()
        return float(w @ ret[valid])

    stock_m = (
        df_m.groupby(['ts_code_x','ym','size_q','bm_q'])['ret']
        .apply(lambda s: (1+s).prod()-1)
        .reset_index()
    )
    mv_last = df_m.sort_values('date').groupby(['ts_code_x','ym'])['total_mv'].last().reset_index()
    stock_m = stock_m.merge(mv_last, on=['ts_code_x','ym'], how='left')

    port_ret = (
        stock_m.groupby(['ym','size_q','bm_q'])
        .apply(vw, include_groups=False)
        .reset_index()
        .rename(columns={0:'ret'})
    )
    port_ret['port'] = port_ret['size_q'].astype(str) + '_' + port_ret['bm_q'].astype(str)
    port_pivot    = port_ret.pivot_table(index='ym', columns='port', values='ret')
    port_excess   = port_pivot.subtract(RF_MONTHLY)
    port_excess.index = port_excess.index.to_timestamp()

    fac_m = (
        df_daily.drop_duplicates(subset=['date'])
        .sort_values('date')
        [['date','ym'] + factor_cols]
        .dropna(subset=factor_cols)
        .groupby('ym')[factor_cols]
        .apply(compound_ret)
        .reset_index()
    )
    fac_m.index = fac_m['ym'].dt.to_timestamp()
    fac_m = fac_m[factor_cols]

    common = port_excess.index.intersection(fac_m.index)
    R = port_excess.loc[common].dropna(axis=1)
    F = fac_m.loc[common].dropna()
    common2 = R.index.intersection(F.index)
    R = R.loc[common2]
    F = F.loc[common2]
    T_actual, N_full = R.shape

    if N_full > N_max:
        col_counts = R.notna().sum().sort_values(ascending=False)
        keep_ports = col_counts.index[:N_max].tolist()
        R = R[keep_ports].dropna()

    return R, F


port_ff5,  fac_ff5  = build_test_portfolios(df_ff5,  FF5_COLS,  T_ff5)
port_ff5n, fac_ff5n = build_test_portfolios(df_ff5n, FF5N_COLS, T_ff5n)


# ══════════════════════════════════════════════════════════════════
# BƯỚC 3: GRS TEST
# ══════════════════════════════════════════════════════════════════

def grs_test(port_excess, factors_m, factor_cols):
    if port_excess is None:
        return None, None, None, None

    common = port_excess.index.intersection(factors_m.index)
    R = port_excess.loc[common].dropna(axis=1)
    F = factors_m.loc[common, factor_cols].dropna()
    common2 = R.index.intersection(F.index)
    R = R.loc[common2]
    F = F.loc[common2]
    R = R.dropna(axis=1)

    T, N = R.shape
    K    = len(factor_cols)

    if T <= N + K + 1:
        N_max = T - K - 2
        if N_max < 3:
            return None, None, None, None
        col_counts = R.notna().sum().sort_values(ascending=False)
        R = R[col_counts.index[:N_max]].dropna()
        T, N = R.shape

    alphas = np.zeros(N)
    resids = np.zeros((T, N))
    X      = sm.add_constant(F.values)
    r2s    = []

    for i, col in enumerate(R.columns):
        y   = R[col].values
        res = sm.OLS(y, X).fit()
        alphas[i]    = res.params[0]
        resids[:, i] = res.resid
        r2s.append(res.rsquared)

    Sigma = np.cov(resids.T) if N > 1 else np.array([[np.var(resids[:,0])]])
    mu    = F.values.mean(axis=0)
    Omega = np.cov(F.values.T) if K > 1 else np.array([[F.values.var()]])

    try:
        Sigma_inv = np.linalg.inv(Sigma)
        Omega_inv = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)
        Omega_inv = np.linalg.pinv(Omega)

    quad_a   = float(alphas @ Sigma_inv @ alphas)
    sharpe_f = float(mu @ Omega_inv @ mu)

    grs_stat = (T/N) * ((T-N-K) / (T-K-1)) * (quad_a / (1 + sharpe_f))
    p_value  = 1 - stats.f.cdf(grs_stat, dfn=N, dfd=T-N-K)

    return grs_stat, p_value, float(np.mean(np.abs(alphas))), float(np.mean(r2s))


g5_stat,  g5_p,  g5_a,  g5_r2  = grs_test(port_ff5,  fac_ff5,  FF5_COLS)
g5n_stat, g5n_p, g5n_a, g5n_r2 = grs_test(port_ff5n, fac_ff5n, FF5N_COLS)


# ══════════════════════════════════════════════════════════════════
# BƯỚC 4: STOCK-LEVEL OLS
# ══════════════════════════════════════════════════════════════════

def run_ols_per_stock(monthly_df, factor_cols, model_name):
    results = []
    for stock, sub in tqdm(monthly_df.groupby('ts_code_x'),
                           desc=f"OLS [{model_name}]", leave=False):
        sub = sub.dropna(subset=['exc_ret_m'] + factor_cols)
        if len(sub) < MIN_OBS:
            continue
        y = sub['exc_ret_m'].values
        X = sm.add_constant(sub[factor_cols].values)
        try:
            res = sm.OLS(y, X).fit()
            row = {
                'ts_code_x': stock,
                'alpha':     res.params[0],
                't_alpha':   res.tvalues[0],
                'p_alpha':   res.pvalues[0],
                'R2':        res.rsquared,
                'adj_R2':    res.rsquared_adj,
                'N_obs':     len(sub)
            }
            for j, c in enumerate(factor_cols):
                row[f'beta_{c}'] = res.params[j+1]
                row[f't_{c}']    = res.tvalues[j+1]
            results.append(row)
        except:
            continue
    return pd.DataFrame(results)


ols_ff5  = run_ols_per_stock(monthly_ff5,  FF5_COLS,  'FF5')
ols_ff5n = run_ols_per_stock(monthly_ff5n, FF5N_COLS, 'FF5-News')


# ══════════════════════════════════════════════════════════════════
# BƯỚC 5: FAMA-MACBETH
# ══════════════════════════════════════════════════════════════════

def fama_macbeth(monthly_df, ols_results, factor_cols):
    beta_src = [f'beta_{c}' for c in factor_cols]
    beta_dst = [f'b_{c}'    for c in factor_cols]

    betas_df = (
        ols_results[['ts_code_x'] + beta_src]
        .copy()
        .rename(columns=dict(zip(beta_src, beta_dst)))
    )

    sub = monthly_df.merge(betas_df, on='ts_code_x', how='inner')
    sub = sub.dropna(subset=['exc_ret_m'] + beta_dst)

    months       = sorted(sub['ym'].unique())
    lambdas_list = []

    for ym in months:
        m_data = sub[sub['ym'] == ym]
        if len(m_data) < len(factor_cols) + 2:
            continue
        y_cs = m_data['exc_ret_m'].values
        X_cs = sm.add_constant(m_data[beta_dst].values)
        try:
            res_cs = sm.OLS(y_cs, X_cs).fit()
            lambdas_list.append(res_cs.params)
        except:
            continue

    if len(lambdas_list) == 0:
        return pd.DataFrame()

    lambdas_arr = np.array(lambdas_list)
    T_fmb       = len(lambdas_arr)
    col_names   = ['intercept'] + factor_cols

    lam_mean = np.nanmean(lambdas_arr, axis=0)
    lam_std  = np.nanstd(lambdas_arr,  axis=0, ddof=1)
    lam_se   = lam_std / np.sqrt(T_fmb)
    lam_t    = lam_mean / (lam_se + 1e-12)
    lam_p    = 2 * stats.t.sf(np.abs(lam_t), df=T_fmb-1)

    return pd.DataFrame({
        'lambda_mean': lam_mean,
        'lambda_se':   lam_se,
        't_stat':      lam_t,
        'p_value':     lam_p,
        'significant': np.abs(lam_t) > 1.96,
    }, index=col_names)


fmb_ff5  = fama_macbeth(monthly_ff5,  ols_ff5,  FF5_COLS)
fmb_ff5n = fama_macbeth(monthly_ff5n, ols_ff5n, FF5N_COLS)


# ══════════════════════════════════════════════════════════════════
# BƯỚC 6: ALPHA COMPARISON + PAIRED T-TEST
# ══════════════════════════════════════════════════════════════════

common_stocks = set(ols_ff5['ts_code_x']) & set(ols_ff5n['ts_code_x'])

alpha_compare = (
    ols_ff5[ols_ff5['ts_code_x'].isin(common_stocks)]
    [['ts_code_x','alpha','R2']]
    .rename(columns={'alpha':'alpha_ff5','R2':'R2_ff5'})
    .merge(
        ols_ff5n[ols_ff5n['ts_code_x'].isin(common_stocks)]
        [['ts_code_x','alpha','R2']]
        .rename(columns={'alpha':'alpha_ff5n','R2':'R2_ff5n'}),
        on='ts_code_x'
    )
)
alpha_compare['abs_alpha_ff5']  = alpha_compare['alpha_ff5'].abs()
alpha_compare['abs_alpha_ff5n'] = alpha_compare['alpha_ff5n'].abs()
alpha_compare['alpha_reduction_pct'] = (
    (alpha_compare['abs_alpha_ff5'] - alpha_compare['abs_alpha_ff5n'])
    / alpha_compare['abs_alpha_ff5'].replace(0, np.nan) * 100
)
alpha_compare['R2_improvement'] = alpha_compare['R2_ff5n'] - alpha_compare['R2_ff5']

# Paired t-test: |α| FF5 - FF5N (positive = FF5N better)
t_a, p_a = stats.ttest_rel(alpha_compare['abs_alpha_ff5'],  alpha_compare['abs_alpha_ff5n'])
# Paired t-test: R² FF5N - FF5 (positive = FF5N better)
t_r, p_r = stats.ttest_rel(alpha_compare['R2_ff5n'], alpha_compare['R2_ff5'])


# ══════════════════════════════════════════════════════════════════
# BƯỚC 7: TABLE 2 — PAPER-STYLE OUTPUT
# ══════════════════════════════════════════════════════════════════

paper_table = pd.DataFrame({
    'Model': [
        'Fama-French 5 Factors',
        'Fama-French 5 Factors with News Effect Factor'
    ],
    'GRS': [
        round(g5_stat,  4) if g5_stat  is not None else 'N/A',
        round(g5n_stat, 4) if g5n_stat is not None else 'N/A',
    ],
    'GRS p-value': [
        f"{g5_p:.3e}"  if g5_p  is not None else 'N/A',
        f"{g5n_p:.3e}" if g5n_p is not None else 'N/A',
    ],
    'Mean |Alpha|': [
        round(ols_ff5['alpha'].abs().mean(),  4),
        round(ols_ff5n['alpha'].abs().mean(), 4),
    ],
    'Mean R²': [
        round(ols_ff5['R2'].mean(),  4),
        round(ols_ff5n['R2'].mean(), 4),
    ]
})

print("\nTable 2: GRS test results and return forecasting evaluation")
print(paper_table.to_string(index=False))


# ══════════════════════════════════════════════════════════════════
# BƯỚC 8: SAVE
# ══════════════════════════════════════════════════════════════════

paper_table.to_csv(os.path.join(OUT_DIR, 'paper_style_table2.csv'), index=False)

latex_table = paper_table.to_latex(
    index=False,
    escape=False,
    caption='GRS test results and return forecasting evaluation results',
    label='tab:grs_results'
)
with open(os.path.join(OUT_DIR, 'paper_style_table2.tex'), 'w') as f:
    f.write(latex_table)

ols_ff5.round(6).to_csv(os.path.join(OUT_DIR, 'ols_results_ff5.csv'), index=False)
ols_ff5n.round(6).to_csv(os.path.join(OUT_DIR, 'ols_results_ff5news.csv'), index=False)
alpha_compare.round(6).to_csv(os.path.join(OUT_DIR, 'alpha_comparison.csv'), index=False)

if len(fmb_ff5) > 0:
    fmb_ff5.round(6).to_csv(os.path.join(OUT_DIR, 'fmb_ff5.csv'))
if len(fmb_ff5n) > 0:
    fmb_ff5n.round(6).to_csv(os.path.join(OUT_DIR, 'fmb_ff5news.csv'))