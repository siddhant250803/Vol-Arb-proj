# ============================================================
# rv_models.py — Realized Volatility Forecasting Models
# ============================================================
"""
Implements three families of RV forecasting models:

    1. HAR-RV   (Heterogeneous Autoregressive — Corsi 2009)
    2. GARCH    (Bollerslev 1986) / EGARCH (Nelson 1991) / GJR-GARCH
    3. Realized-GARCH  (Hansen, Huang & Shek 2012)

Each model exposes:
    - fit(train_data)   → fitted model object
    - forecast(model, steps)  → h-step-ahead volatility forecast
    - rolling_forecast(data, train_window, horizon)
        → out-of-sample forecasts aligned to the feature table
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from arch import arch_model

from src.config import (
    HAR_LAGS,
    GARCH_P,
    GARCH_Q,
    RV_FORECAST_HORIZON,
    ANNUALISATION_FACTOR,
    TRADING_DAYS_PER_YEAR,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ════════════════════════════════════════════════════════════
# 1.  HAR-RV  (Heterogeneous Autoregressive Model)
# ════════════════════════════════════════════════════════════

class HARRV:
    """
    The HAR-RV model of Corsi (2009):

        RV_{t→t+h} = β₀ + β₁·RV_daily + β₂·RV_weekly + β₃·RV_monthly + ε

    where RV_daily/weekly/monthly are look-back averages.
    """

    def __init__(self, lags=None):
        """
        Parameters
        ----------
        lags : list of int
            Look-back windows for the HAR regressors (default: [1, 5, 22]).
        """
        self.lags = lags or HAR_LAGS
        self.model_ = None
        self.coefficients_ = None

    def _build_features(self, rv_series):
        """
        Construct lagged RV features for the HAR model.

        Parameters
        ----------
        rv_series : pd.Series
            Daily realised variance (annualised).

        Returns
        -------
        pd.DataFrame
            Columns: rv_lag_{lag} for each lag in self.lags.
        """
        df = pd.DataFrame({"rv": rv_series.values}, index=rv_series.index)
        for lag in self.lags:
            df[f"rv_lag_{lag}"] = (
                df["rv"].rolling(window=lag, min_periods=lag).mean()
            )
        return df.dropna()

    def fit(self, rv_series, fwd_rv_series):
        """
        Fit the HAR-RV model via OLS.

        Parameters
        ----------
        rv_series : pd.Series
            Historical daily realised variance.
        fwd_rv_series : pd.Series
            Forward realised variance (target).

        Returns
        -------
        self
        """
        features = self._build_features(rv_series)
        # Align target
        common = features.index.intersection(fwd_rv_series.index)
        X = features.loc[common, [f"rv_lag_{l}" for l in self.lags]]
        y = fwd_rv_series.loc[common]

        X = add_constant(X)
        self.model_ = OLS(y, X).fit()
        self.coefficients_ = self.model_.params
        return self

    def predict(self, rv_series):
        """
        Generate in-sample or out-of-sample forecasts.

        Parameters
        ----------
        rv_series : pd.Series
            Historical daily realised variance.

        Returns
        -------
        pd.Series
            Forecast of forward realised variance.
        """
        features = self._build_features(rv_series)
        X = features[[f"rv_lag_{l}" for l in self.lags]]
        X = add_constant(X)
        pred = self.model_.predict(X)
        return pd.Series(pred, index=features.index, name="har_rv_forecast")

    def summary(self):
        """Print regression summary."""
        if self.model_ is not None:
            print(self.model_.summary())


def har_rv_rolling_forecast(rv_series, fwd_rv_series, train_window=504):
    """
    Expanding-window HAR-RV forecast.

    Parameters
    ----------
    rv_series : pd.Series
        Daily realised variance (indexed by integer position or date).
    fwd_rv_series : pd.Series
        Forward realised variance targets.
    train_window : int
        Minimum number of observations for initial training.

    Returns
    -------
    pd.Series
        Out-of-sample forecasts aligned to the date index.
    """
    model = HARRV()
    forecasts = {}

    common = rv_series.index.intersection(fwd_rv_series.index)
    rv_aligned = rv_series.loc[common]
    fwd_aligned = fwd_rv_series.loc[common]

    n = len(rv_aligned)
    for t in range(train_window, n):
        train_rv = rv_aligned.iloc[:t]
        train_fwd = fwd_aligned.iloc[:t]

        try:
            model.fit(train_rv, train_fwd)
            # Predict at time t using features available at t
            last_rv = rv_aligned.iloc[: t + 1]
            pred = model.predict(last_rv)
            if len(pred) > 0:
                forecasts[rv_aligned.index[t]] = pred.iloc[-1]
        except Exception:
            continue

    out = pd.Series(forecasts, name="har_rv_forecast")
    print(f"[rv_models] HAR-RV rolling forecast: {len(out)} predictions.")
    return out


# ════════════════════════════════════════════════════════════
# 2.  GARCH Family Models
# ════════════════════════════════════════════════════════════

def fit_garch(returns, model_type="GARCH", p=None, q=None):
    """
    Fit a GARCH-family model to a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily log-returns (not percentage, will be scaled internally).
    model_type : str
        One of "GARCH", "EGARCH", "GJR" (GJR-GARCH).
    p : int
        Lag order for the variance equation.
    q : int
        Lag order for the ARCH term.

    Returns
    -------
    arch model result
        Fitted model with `forecast()` capability.
    """
    p = p or GARCH_P
    q = q or GARCH_Q

    # arch package expects returns scaled to percentage
    scaled = returns * 100.0

    vol_model = model_type if model_type != "GJR" else "GARCH"
    o = 1 if model_type == "GJR" else 0

    if model_type == "EGARCH":
        am = arch_model(
            scaled.dropna(),
            mean="Constant",
            vol="EGARCH",
            p=p,
            q=q,
        )
    else:
        am = arch_model(
            scaled.dropna(),
            mean="Constant",
            vol=vol_model,
            p=p,
            o=o,
            q=q,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = am.fit(disp="off", show_warning=False)
    return result


def garch_rolling_forecast(
    returns,
    model_type="GARCH",
    train_window=504,
    forecast_horizon=22,
):
    """
    Expanding-window GARCH variance forecast.

    For each date t (after the training window), fits GARCH on returns[:t]
    and forecasts the variance over the next ``forecast_horizon`` days.

    Parameters
    ----------
    returns : pd.Series
        Daily log-returns indexed by date.
    model_type : str
        "GARCH", "EGARCH", or "GJR".
    train_window : int
        Minimum training observations.
    forecast_horizon : int
        Number of days ahead to forecast.

    Returns
    -------
    pd.Series
        Annualised variance forecasts indexed by date.
    """
    n = len(returns)
    forecasts = {}

    # Re-fit every 5 days for speed (use last model in between)
    refit_interval = 5

    for t in range(train_window, n, refit_interval):
        try:
            train = returns.iloc[:t]
            result = fit_garch(train, model_type=model_type)
            fcast = result.forecast(horizon=forecast_horizon)

            # Average variance over the forecast horizon, annualise
            # fcast.variance gives variance in (pct return)² units
            avg_var = fcast.variance.iloc[-1].mean()
            # Convert from percentage² to decimal² and annualise
            ann_var = (avg_var / 10000.0) * ANNUALISATION_FACTOR

            date = returns.index[t]
            forecasts[date] = ann_var

            # Fill in skipped dates with same forecast
            for j in range(1, min(refit_interval, n - t)):
                if t + j < n:
                    forecasts[returns.index[t + j]] = ann_var
        except Exception:
            continue

    out = pd.Series(forecasts, name=f"{model_type.lower()}_forecast")
    print(f"[rv_models] {model_type} rolling forecast: {len(out)} predictions.")
    return out


# ════════════════════════════════════════════════════════════
# 3.  Convenience: Run All Models
# ════════════════════════════════════════════════════════════

def run_all_rv_models(feature_df, train_window=504):
    """
    Run HAR-RV, GARCH, EGARCH, and GJR-GARCH on the feature table
    and return a DataFrame of forecasts.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Master feature table with columns:
        date, log_return, rv_monthly, fwd_rv.
    train_window : int
        Minimum observations for model training.

    Returns
    -------
    pd.DataFrame
        Columns: date, har_rv_forecast, garch_forecast,
                 egarch_forecast, gjr_forecast
    """
    print("\n" + "=" * 60)
    print("  RUNNING RV FORECAST MODELS")
    print("=" * 60)

    df = feature_df.set_index("date").sort_index()

    # ── HAR-RV ─────────────────────────────────────────────
    har_fcast = har_rv_rolling_forecast(
        df["rv_monthly"],
        df["fwd_rv"].dropna(),
        train_window=train_window,
    )

    # ── GARCH Family ───────────────────────────────────────
    garch_fcast = garch_rolling_forecast(
        df["log_return"], "GARCH", train_window
    )
    egarch_fcast = garch_rolling_forecast(
        df["log_return"], "EGARCH", train_window
    )
    gjr_fcast = garch_rolling_forecast(
        df["log_return"], "GJR", train_window
    )

    # ── Merge ──────────────────────────────────────────────
    result = pd.DataFrame(index=df.index)
    result["har_rv_forecast"] = har_fcast
    result["garch_forecast"] = garch_fcast
    result["egarch_forecast"] = egarch_fcast
    result["gjr_forecast"] = gjr_fcast

    # Composite forecast (equal-weight average, skip missing models)
    fcast_cols = [
        "har_rv_forecast",
        "garch_forecast",
        "egarch_forecast",
        "gjr_forecast",
    ]
    # Only average over models that have at least some data
    active_cols = [c for c in fcast_cols if result[c].notna().sum() > 0]
    if active_cols:
        result["composite_rv_forecast"] = result[active_cols].mean(axis=1)
    else:
        result["composite_rv_forecast"] = np.nan

    result = result.reset_index()

    n_any = result[fcast_cols].notna().any(axis=1).sum()
    print(f"\n[rv_models] Combined forecasts: {n_any} rows "
          f"with at least one model. Active models: {active_cols}")
    return result
