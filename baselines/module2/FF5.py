import numpy as np
import pandas as pd
import os
import glob
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 0: LOAD DATA
# ══════════════════════════════════════════════════════════════════

dfs = []
paths = [
    r'/kaggle/working/stock_daily/stock_daily1',
    r'/kaggle/working/stock_daily/stock_daily2',
    r'/kaggle/working/stock_daily/stock_daily3',
]
for p in paths:
    if not os.path.isdir(p):
        continue
    for f in sorted(glob.glob(os.path.join(p, '*.csv'))):
        try:
            dfs.append(pd.read_csv(f, sep='\t', encoding='utf-8'))
        except:
            pass

df_raw = pd.concat(dfs, ignore_index=True)

d1_train = pd.read_csv("/kaggle/input/datasets/phmhngtrang/module2-input/roberta_srl_sdpg_features_train.csv")
d1_val   = pd.read_csv("/kaggle/input/datasets/phmhngtrang/module2-input/roberta_srl_sdpg_features_val.csv")
d1_test  = pd.read_csv("/kaggle/input/datasets/phmhngtrang/module2-input/roberta_srl_sdpg_features_test.csv")

df_news_all = pd.concat([d1_train, d1_val, d1_test], ignore_index=True)
df_news_all['trade_date'] = pd.to_datetime(df_news_all['trade_date'], errors='coerce')


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1: CLEAN & NORMALIZE
# ══════════════════════════════════════════════════════════════════

df = df_raw.copy()

if 'close_x' in df.columns:
    df.rename(columns={'close_x': 'close'}, inplace=True)

df['date']  = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df['ym']    = df['date'].dt.to_period('M')

start = max(df['date'].min(), df_news_all['trade_date'].min())
end   = min(df['date'].max(), df_news_all['trade_date'].max())
df = df[(df['date'] >= start) & (df['date'] <= end)]

df = df.dropna(subset=['total_mv', 'close'])
df = df[df['total_mv'] > 0]
df = df[df['close'] > 0]

RF_ANNUAL = 0.0225
RF_DAILY  = RF_ANNUAL / 252

df['ret']        = df['pct_chg'] / 100.0
df['ret_excess'] = df['ret'] - RF_DAILY
df['bm']         = np.where(df['pb'] > 0, 1.0 / df['pb'], np.nan)
df['op']         = np.where(df['pe_ttm'] > 0, 1.0 / df['pe_ttm'], np.nan)

def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))

df['bm'] = winsorize(df['bm'].dropna()).reindex(df.index)
df['op'] = winsorize(df['op'].dropna()).reindex(df.index)


# ══════════════════════════════════════════════════════════════════
# BƯỚC 2: INVESTMENT PROXY (12M MV growth)
# ══════════════════════════════════════════════════════════════════

monthly_mv = (
    df.sort_values('date')
    .groupby(['ts_code_x', 'ym'])
    .last()[['total_mv']]
    .reset_index()
)
monthly_mv['ym_dt']          = monthly_mv['ym'].dt.to_timestamp()
monthly_mv                   = monthly_mv.sort_values(['ts_code_x', 'ym_dt'])
monthly_mv['total_mv_lag12'] = monthly_mv.groupby('ts_code_x')['total_mv'].shift(12)
monthly_mv['inv'] = (
    (monthly_mv['total_mv'] - monthly_mv['total_mv_lag12'])
    / monthly_mv['total_mv_lag12'].abs()
)
monthly_mv['inv'] = winsorize(monthly_mv['inv'].dropna()).reindex(monthly_mv.index)

df = df.merge(monthly_mv[['ts_code_x', 'ym', 'inv']], on=['ts_code_x', 'ym'], how='left')


# ══════════════════════════════════════════════════════════════════
# BƯỚC 3: SORT PORTFOLIOS — FAMA-FRENCH 2×3 (per month)
# ══════════════════════════════════════════════════════════════════

eom = (
    df.sort_values('date')
    .groupby(['ts_code_x', 'ym'])
    .last()
    .reset_index()
    [['ts_code_x', 'ym', 'total_mv', 'bm', 'op', 'inv']]
)
eom = eom.dropna(subset=['total_mv', 'bm'])
eom['ym_next'] = eom['ym'] + 1

def assign_groups(grp, var, lo_q=0.30, hi_q=0.70):
    lo = grp[var].quantile(lo_q)
    hi = grp[var].quantile(hi_q)
    result = pd.Series('M', index=grp.index)
    result[grp[var] <= lo] = 'L'
    result[grp[var] >= hi] = 'H'
    return result

def assign_size(grp):
    med = grp['total_mv'].median()
    return pd.Series(np.where(grp['total_mv'] <= med, 'S', 'B'), index=grp.index)

tqdm.pandas(desc="Sorting portfolios")

eom['size_grp'] = eom.groupby('ym', group_keys=False).apply(assign_size)
eom['bm_grp']   = eom.groupby('ym', group_keys=False).apply(lambda g: assign_groups(g, 'bm'))
eom['op_grp']   = eom.groupby('ym', group_keys=False).apply(
    lambda g: assign_groups(g[g['op'].notna()], 'op')
    if g['op'].notna().sum() > 10
    else pd.Series('M', index=g.index)
)
eom['inv_grp']  = eom.groupby('ym', group_keys=False).apply(
    lambda g: assign_groups(g[g['inv'].notna()], 'inv')
    if g['inv'].notna().sum() > 10
    else pd.Series('M', index=g.index)
)

sort_df = eom[['ts_code_x', 'ym_next', 'size_grp', 'bm_grp', 'op_grp', 'inv_grp']].rename(
    columns={'ym_next': 'ym'}
)
df = df.merge(sort_df, on=['ts_code_x', 'ym'], how='left')


# ══════════════════════════════════════════════════════════════════
# BƯỚC 4: VW PORTFOLIO RETURNS PER DAY
# ══════════════════════════════════════════════════════════════════

def vw_ret(grp):
    mv  = grp['total_mv'].values
    ret = grp['ret'].values
    valid = np.isfinite(mv) & np.isfinite(ret) & (mv > 0)
    if valid.sum() == 0:
        return np.nan
    w = mv[valid] / mv[valid].sum()
    return float(w @ ret[valid])

hml_port = (
    df.dropna(subset=['size_grp', 'bm_grp'])
    .groupby(['date', 'size_grp', 'bm_grp'])
    .apply(vw_ret, include_groups=False)
    .reset_index()
)
hml_port.columns = ['date', 'size_grp', 'bm_grp', 'vw_ret']

rmw_port = (
    df.dropna(subset=['size_grp', 'op_grp'])
    .groupby(['date', 'size_grp', 'op_grp'])
    .apply(vw_ret, include_groups=False)
    .reset_index()
)
rmw_port.columns = ['date', 'size_grp', 'op_grp', 'vw_ret']

cma_port = (
    df.dropna(subset=['size_grp', 'inv_grp'])
    .groupby(['date', 'size_grp', 'inv_grp'])
    .apply(vw_ret, include_groups=False)
    .reset_index()
)
cma_port.columns = ['date', 'size_grp', 'inv_grp', 'vw_ret']


# ══════════════════════════════════════════════════════════════════
# BƯỚC 5: COMPUTE 5 FACTORS
# ══════════════════════════════════════════════════════════════════

def pivot_port(port_df, col1, col2):
    return port_df.pivot_table(index='date', columns=[col1, col2], values='vw_ret')

def safe_get(piv, key, default=np.nan):
    try:
        return piv[key]
    except KeyError:
        return pd.Series(default, index=piv.index)

mkt = (
    df.groupby('date')
    .apply(lambda g: pd.Series({'MKT': vw_ret(g)}))
    .reset_index()
)
mkt['MKT_RF'] = mkt['MKT'] - RF_DAILY

hml_piv = pivot_port(hml_port, 'size_grp', 'bm_grp')
rmw_piv = pivot_port(rmw_port, 'size_grp', 'op_grp')
cma_piv = pivot_port(cma_port, 'size_grp', 'inv_grp')

SMB_bm = (
    (safe_get(hml_piv, ('S','H')) + safe_get(hml_piv, ('S','M')) + safe_get(hml_piv, ('S','L'))) / 3
    - (safe_get(hml_piv, ('B','H')) + safe_get(hml_piv, ('B','M')) + safe_get(hml_piv, ('B','L'))) / 3
)
SMB_op = (
    (safe_get(rmw_piv, ('S','H')) + safe_get(rmw_piv, ('S','M')) + safe_get(rmw_piv, ('S','L'))) / 3
    - (safe_get(rmw_piv, ('B','H')) + safe_get(rmw_piv, ('B','M')) + safe_get(rmw_piv, ('B','L'))) / 3
)
SMB_inv = (
    (safe_get(cma_piv, ('S','H')) + safe_get(cma_piv, ('S','M')) + safe_get(cma_piv, ('S','L'))) / 3
    - (safe_get(cma_piv, ('B','H')) + safe_get(cma_piv, ('B','M')) + safe_get(cma_piv, ('B','L'))) / 3
)
SMB = (SMB_bm + SMB_op + SMB_inv) / 3

HML = (
    (safe_get(hml_piv, ('S','H')) + safe_get(hml_piv, ('B','H'))) / 2
    - (safe_get(hml_piv, ('S','L')) + safe_get(hml_piv, ('B','L'))) / 2
)
RMW = (
    (safe_get(rmw_piv, ('S','H')) + safe_get(rmw_piv, ('B','H'))) / 2
    - (safe_get(rmw_piv, ('S','L')) + safe_get(rmw_piv, ('B','L'))) / 2
)
CMA = (
    (safe_get(cma_piv, ('S','L')) + safe_get(cma_piv, ('B','L'))) / 2
    - (safe_get(cma_piv, ('S','H')) + safe_get(cma_piv, ('B','H'))) / 2
)

factors_daily = pd.DataFrame({
    'MKT_RF': mkt.set_index('date')['MKT_RF'],
    'SMB':    SMB,
    'HML':    HML,
    'RMW':    RMW,
    'CMA':    CMA,
}).dropna(how='all')

factors_daily.index.name = 'date'
factors_daily = factors_daily.sort_index()

print(factors_daily.describe().round(6))
print(f"\nAnnualized factor means (×252):")
print((factors_daily.mean() * 252).round(4))


# ══════════════════════════════════════════════════════════════════
# BƯỚC 6: MONTHLY AGGREGATION
# ══════════════════════════════════════════════════════════════════

def compound_ret(s):
    return (1 + s).prod() - 1

factors_daily_reset = factors_daily.reset_index()
factors_daily_reset['ym'] = factors_daily_reset['date'].dt.to_period('M')

factors_monthly = (
    factors_daily_reset
    .groupby('ym')[['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']]
    .apply(compound_ret)
)
factors_monthly.index      = factors_monthly.index.to_timestamp()
factors_monthly.index.name = 'date'


# ══════════════════════════════════════════════════════════════════
# BƯỚC 7: MODULE2 FORMAT OUTPUT
# ══════════════════════════════════════════════════════════════════

stock_out = df[['ts_code_x', 'date', 'ret', 'ret_excess', 'total_mv']].copy()
stock_out = stock_out.merge(factors_daily.reset_index(), on='date', how='left')
stock_out.rename(columns={'ret_excess': 'excess_return'}, inplace=True)
stock_out['date_norm'] = stock_out['date'].dt.strftime('%Y-%m-%d')

out_cols         = ['ts_code_x', 'date_norm', 'excess_return', 'MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
df_module2_rebuilt = stock_out[out_cols].dropna(subset=['MKT_RF', 'SMB', 'HML', 'CMA'])


# ══════════════════════════════════════════════════════════════════
# BƯỚC 8: SAVE
# ══════════════════════════════════════════════════════════════════

out_dir = r'/kaggle/working/outputs'
os.makedirs(out_dir, exist_ok=True)

factors_daily.to_csv(os.path.join(out_dir, 'ff5_factors_daily.csv'))
factors_monthly.to_csv(os.path.join(out_dir, 'ff5_factors_monthly.csv'))
df_module2_rebuilt.to_csv(os.path.join(out_dir, 'module2_data_rebuilt.csv'), index=False)