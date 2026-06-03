import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from arch import arch_model
import warnings


class EGARCHVaRModel:
    def __init__(self, config):
        self.alpha = config['var']['confidence_level']
        self.min_obs = config['var']['min_samples']
        self.max_zero_ratio = config['var'].get('max_zero_ratio', 0.1)

        self.p = config['egarch']['p']
        self.q = config['egarch']['q']
        self.dist = config['egarch']['dist']
        self.scale = config['egarch']['scale']
        self.window = config['egarch']['rolling_window']

    def check_data(self, returns):
        n = len(returns)
        if n < self.min_obs:
            return False, f"Too few observations ({n})"
        zero_ratio = np.sum(returns == 0) / n
        if zero_ratio > self.max_zero_ratio:
            return False, f"Too many zero returns ({zero_ratio:.1%})"
        if np.std(returns) < 1e-10:
            return False, "Near-constant series"
        if np.any(~np.isfinite(returns)):
            return False, "NaN or Inf values"
        return True, "OK"

    def _get_z_alpha(self, res=None):
        """
        Lấy z_alpha đúng theo phân phối đã dùng để fit model.
        Nếu dist='t', lấy bậc tự do từ kết quả fit.
        Nếu fallback (res=None), dùng normal.
        """
        if res is not None and self.dist == 't':
            try:
                # arch lưu nu (degrees of freedom) trong params
                nu = res.params.get('nu', None)
                if nu is not None and np.isfinite(nu) and nu > 2:
                    return float(student_t.ppf(self.alpha, df=nu))
            except Exception:
                pass
        # default: normal
        return float(norm.ppf(self.alpha))

    def compute_var(self, stock_df):
        returns = stock_df['return'].values
        returns = np.clip(returns, -0.5, 0.5)

        valid, msg = self.check_data(returns)
        if not valid:
            raise ValueError(msg)

        returns_scaled = returns * self.scale
        res = None

        # -------- LAYER 1: EGARCH --------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = arch_model(
                    returns_scaled, vol='EGARCH',
                    p=self.p, q=self.q,
                    dist=self.dist, rescale=True
                )
                res = model.fit(
                    disp='off',
                    options={'maxiter': 2000, 'ftol': 1e-6}
                )
                if not res.converged:
                    raise ValueError("EGARCH not converged")
                if not np.all(np.isfinite(res.conditional_volatility)):
                    raise ValueError("Invalid volatility")
            except Exception:
                res = None

        # -------- LAYER 2: GARCH --------
        if res is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = arch_model(
                        returns_scaled, vol='Garch',
                        p=self.p, q=self.q,
                        dist=self.dist, rescale=True
                    )
                    res = model.fit(disp='off', options={'maxiter': 2000})
                    if not res.converged:
                        raise ValueError()
                    if not np.all(np.isfinite(res.conditional_volatility)):
                        raise ValueError()
                except Exception:
                    res = None

        # -------- LAYER 3: rolling std --------
        if res is not None:
            sigma_last = res.conditional_volatility[-1] / self.scale
        else:
            sigma_last = (
                pd.Series(returns).rolling(self.window).std().dropna().iloc[-1]
            )

        mu_last = (
            pd.Series(returns).rolling(self.window).mean().dropna().iloc[-1]
        )

        # ✅ z_alpha đúng phân phối (t hoặc normal tùy fit)
        z_alpha = self._get_z_alpha(res)  # âm, vd: -1.645 (normal) hoặc -1.7 (t, nu~10)

        # return-based VaR: âm = lỗ
        predicted_var = float(mu_last + sigma_last * z_alpha)
        actual_var = float(np.quantile(returns, self.alpha))

        return predicted_var, actual_var