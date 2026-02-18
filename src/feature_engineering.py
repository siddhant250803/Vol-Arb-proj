# ============================================================
# feature_engineering.py — IV Measures & Realized Volatility
# ============================================================
"""
Responsible for:
    1. ATM implied volatility extraction (constant-maturity interpolation)
    2. Model-free implied variance (Carr-Madan variance swap)
    3. Realized variance / volatility computation
    4. Bipower variation (jump-robust RV)
    5. Event-flag features (FOMC, CPI, NFP windows)

These features feed into the forecasting models and signal generators.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.config import (
    COL,
    ATM_DELTA_BAND,
    CONSTANT_MATURITY_DAYS,
    VARIANCE_SWAP_STRIKE_BAND,
    TRADING_DAYS_PER_YEAR,
    RV_HORIZONS,
    ANNUALISATION_FACTOR,
    FOMC_DATES,
)


# ════════════════════════════════════════════════════════════
# SECTION A:  IMPLIED VOLATILITY MEASURES
# ════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────
# A1.  ATM Implied Volatility (per date)
# ────────────────────────────────────────────────────────────
def compute_atm_iv(options_df):
    """
    For each trading date, extract the ATM implied volatility by:
      1. Selecting options near-the-money (|delta| in [0.40, 0.60]).
      2. Averaging call and put IV weighted by inverse distance to |delta|=0.50.
      3. Interpolating across expiries to a constant 30-day maturity.

    Parameters
    ----------
    options_df : pd.DataFrame
        Cleaned options with columns: date, dte, delta, impl_volatility, cp_flag.

    Returns
    -------
    pd.DataFrame
        Columns: date, atm_iv_30d
        One row per trading day.
    """
    df = options_df.copy()
    df["abs_delta"] = df[COL["delta"]].abs()

    # Keep near-ATM options
    atm_mask = (df["abs_delta"] >= 0.50 - ATM_DELTA_BAND) & (
        df["abs_delta"] <= 0.50 + ATM_DELTA_BAND
    )
    atm = df.loc[atm_mask].copy()

    # Weight by closeness to |delta| = 0.50
    atm["w"] = 1.0 / (1e-6 + (atm["abs_delta"] - 0.50).abs())

    # Weighted average IV per (date, dte)
    def _wavg(g):
        return np.average(g[COL["iv"]], weights=g["w"])

    grouped_raw = atm.groupby([COL["date"], "dte"]).apply(
        _wavg, include_groups=False,
    )
    grouped = grouped_raw.reset_index()
    # The value column may be named 0 or something else — rename reliably
    grouped.columns = ["date", "dte", "atm_iv"]

    # Interpolate to constant 30-day maturity per date
    results = []
    for date, grp in grouped.groupby("date"):
        grp = grp.sort_values("dte")
        if len(grp) < 2:
            # Cannot interpolate with fewer than 2 tenors — use single value
            if len(grp) == 1:
                results.append({"date": date, "atm_iv_30d": grp["atm_iv"].iloc[0]})
            continue
        try:
            f = interp1d(
                grp["dte"].values,
                grp["atm_iv"].values,
                kind="linear",
                fill_value="extrapolate",
            )
            iv_30 = float(f(CONSTANT_MATURITY_DAYS))
            if iv_30 > 0:
                results.append({"date": date, "atm_iv_30d": iv_30})
        except Exception:
            continue

    if not results:
        print("[features] ATM IV (30d): 0 dates — no valid ATM options found.")
        return pd.DataFrame(columns=["date", "atm_iv_30d"])

    out = pd.DataFrame(results)
    out["date"] = pd.to_datetime(out["date"])
    print(f"[features] ATM IV (30d): {len(out)} dates computed.")
    return out


# ────────────────────────────────────────────────────────────
# A2.  Model-Free Implied Variance (Variance Swap)
# ────────────────────────────────────────────────────────────
def compute_model_free_iv(options_df, rf_series=None):
    """
    Construct a synthetic variance-swap strike using the discrete
    Carr-Madan approximation:

        σ²_IV ≈ (2/T) Σ (ΔK / K²) · e^{rT} · Q(K)

    where Q(K) = OTM option mid-price (put if K < F, call if K ≥ F),
    and the sum runs over strikes within ±5% of forward/spot.

    Parameters
    ----------
    options_df : pd.DataFrame
        Cleaned options data.
    rf_series : pd.Series, optional
        Risk-free rate indexed by date.  If None, uses r = 0.

    Returns
    -------
    pd.DataFrame
        Columns: date, dte, mfiv  (model-free implied variance, annualised)
    """
    df = options_df.copy()
    df["mid"] = (df[COL["bid"]] + df[COL["ask"]]) / 2.0

    results = []
    for (date, dte), grp in df.groupby([COL["date"], "dte"]):
        spot = grp[COL["spot"]].iloc[0]
        T = dte / TRADING_DAYS_PER_YEAR
        if T <= 0:
            continue

        # Risk-free rate for this date
        r = 0.0
        if rf_series is not None and date in rf_series.index:
            r = rf_series.loc[date]

        # Forward price (use data column if available, else approximate)
        if COL["forward"] in grp.columns and grp[COL["forward"]].notna().any():
            F = grp[COL["forward"]].dropna().iloc[0]
        else:
            F = spot * np.exp(r * T)

        # Filter to strikes within ±5% of forward
        lo = F * (1.0 - VARIANCE_SWAP_STRIKE_BAND)
        hi = F * (1.0 + VARIANCE_SWAP_STRIKE_BAND)
        sub = grp[(grp["strike"] >= lo) & (grp["strike"] <= hi)].copy()
        if len(sub) < 4:
            continue

        # Separate OTM: puts below F, calls at/above F
        puts = sub[(sub[COL["cp_flag"]] == "P") & (sub["strike"] < F)]
        calls = sub[(sub[COL["cp_flag"]] == "C") & (sub["strike"] >= F)]
        otm = pd.concat([puts, calls]).sort_values("strike")

        if len(otm) < 3:
            continue

        strikes = otm["strike"].values
        prices = otm["mid"].values

        # Discrete Carr-Madan summation
        variance = 0.0
        for i in range(len(strikes)):
            K = strikes[i]
            Q = prices[i]
            # ΔK: use midpoint spacing
            if i == 0:
                dK = strikes[1] - strikes[0]
            elif i == len(strikes) - 1:
                dK = strikes[-1] - strikes[-2]
            else:
                dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

            variance += (dK / (K ** 2)) * np.exp(r * T) * Q

        mfiv = (2.0 / T) * variance  # annualised implied variance

        if mfiv > 0:
            results.append({
                "date": date,
                "dte": dte,
                "mfiv": mfiv,
                "mfiv_vol": np.sqrt(mfiv),  # implied vol from var swap
            })

    out = pd.DataFrame(results)
    if len(out) > 0:
        out["date"] = pd.to_datetime(out["date"])
    print(f"[features] Model-free IV: {len(out)} (date, dte) pairs computed.")
    return out


def compute_mfiv_30d(mfiv_df):
    """
    Interpolate model-free implied variance to a constant 30-day maturity.

    Parameters
    ----------
    mfiv_df : pd.DataFrame
        Output of ``compute_model_free_iv()``.

    Returns
    -------
    pd.DataFrame
        Columns: date, mfiv_30d, mfiv_vol_30d
    """
    results = []
    for date, grp in mfiv_df.groupby("date"):
        grp = grp.sort_values("dte")
        if len(grp) < 2:
            continue
        try:
            f_var = interp1d(
                grp["dte"].values,
                grp["mfiv"].values,
                kind="linear",
                fill_value="extrapolate",
            )
            var_30 = float(f_var(CONSTANT_MATURITY_DAYS))
            if var_30 > 0:
                results.append({
                    "date": date,
                    "mfiv_30d": var_30,
                    "mfiv_vol_30d": np.sqrt(var_30),
                })
        except Exception:
            continue

    out = pd.DataFrame(results)
    if len(out) > 0:
        out["date"] = pd.to_datetime(out["date"])
    print(f"[features] MFIV 30d: {len(out)} dates interpolated.")
    return out


# ════════════════════════════════════════════════════════════
# SECTION B:  REALIZED VOLATILITY MEASURES
# ════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────
# B1.  Standard Realized Variance
# ────────────────────────────────────────────────────────────
def compute_realized_variance(spx_df, windows=None):
    """
    Compute rolling realized variance at multiple horizons.

    RV_t(h) = (252/h) * Σ_{i=0}^{h-1} r_{t-i}²

    Parameters
    ----------
    spx_df : pd.DataFrame
        Must have columns: date, log_return.
    windows : dict, optional
        {label: window_size} mapping. Defaults to config.RV_HORIZONS.

    Returns
    -------
    pd.DataFrame
        Columns: date, rv_daily, rv_weekly, rv_monthly
        (annualised realised variance)
    """
    windows = windows or RV_HORIZONS
    df = spx_df[["date", "log_return"]].copy()
    df["return_sq"] = df["log_return"] ** 2

    for label, w in windows.items():
        col = f"rv_{label}"
        df[col] = (
            df["return_sq"]
            .rolling(window=w, min_periods=w)
            .sum()
            * (ANNUALISATION_FACTOR / w)
        )

    # Also compute realised *volatility* (sqrt of variance)
    for label in windows:
        df[f"rvol_{label}"] = np.sqrt(df[f"rv_{label}"])

    df = df.drop(columns=["return_sq"]).dropna()
    print(f"[features] Realized variance: {len(df)} rows for {list(windows.keys())}.")
    return df


# ────────────────────────────────────────────────────────────
# B2.  Bipower Variation (jump-robust)
# ────────────────────────────────────────────────────────────
def compute_bipower_variation(spx_df, window=22):
    """
    Bipower variation: a jump-robust estimator of integrated variance.

    BV_t(h) = (π/2) * (252/h) * Σ_{i=1}^{h-1} |r_{t-i}| · |r_{t-i-1}|

    Parameters
    ----------
    spx_df : pd.DataFrame
        Must have columns: date, log_return.
    window : int
        Look-back window in trading days.

    Returns
    -------
    pd.DataFrame
        Columns: date, bv_monthly (annualised bipower variation)
    """
    df = spx_df[["date", "log_return"]].copy()
    abs_r = df["log_return"].abs()
    bv_raw = abs_r * abs_r.shift(1)
    df["bv_monthly"] = (
        bv_raw.rolling(window=window, min_periods=window).sum()
        * (np.pi / 2)
        * (ANNUALISATION_FACTOR / window)
    )
    df = df[["date", "bv_monthly"]].dropna()
    print(f"[features] Bipower variation: {len(df)} rows (window={window}).")
    return df


# ────────────────────────────────────────────────────────────
# B3.  Forward-Realised Variance  (label for forecasting)
# ────────────────────────────────────────────────────────────
def compute_forward_rv(spx_df, horizon=22):
    """
    Compute the *future* realised variance over the next ``horizon`` days.
    This is the target variable for RV forecast models.

    Parameters
    ----------
    spx_df : pd.DataFrame
        Must have columns: date, log_return.
    horizon : int
        Number of trading days forward.

    Returns
    -------
    pd.DataFrame
        Columns: date, fwd_rv  (annualised)
    """
    df = spx_df[["date", "log_return"]].copy()
    df["return_sq"] = df["log_return"] ** 2
    df["fwd_rv"] = (
        df["return_sq"]
        .shift(-horizon)               # ← look *forward*
        .rolling(window=horizon, min_periods=horizon)
        .sum()
        .shift(-(horizon - 1))          # align to current date
    )
    # Annualise
    df["fwd_rv"] = df["fwd_rv"] * (ANNUALISATION_FACTOR / horizon)
    df["fwd_rvol"] = np.sqrt(df["fwd_rv"].clip(lower=0))

    # Simpler approach: shift squared returns backward
    # Actually, compute sum of next `horizon` squared returns from date t
    return_sq = df["log_return"].values ** 2
    n = len(return_sq)
    fwd_rv = np.full(n, np.nan)
    for i in range(n - horizon):
        fwd_rv[i] = return_sq[i + 1: i + 1 + horizon].sum() * (ANNUALISATION_FACTOR / horizon)

    df["fwd_rv"] = fwd_rv
    df["fwd_rvol"] = np.sqrt(np.maximum(fwd_rv, 0))
    df = df[["date", "fwd_rv", "fwd_rvol"]].dropna()
    print(f"[features] Forward RV ({horizon}d): {len(df)} rows.")
    return df


# ════════════════════════════════════════════════════════════
# SECTION C:  EVENT FLAGS
# ════════════════════════════════════════════════════════════

def add_event_flags(df, date_col="date"):
    """
    Add binary indicators for major macro-event windows.

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame with a date column.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with added columns:
        fomc_window (1 if within ±1 day of FOMC date)
    """
    fomc = pd.to_datetime(FOMC_DATES)
    out = df.copy()

    # Mark dates within ±7 calendar days of an FOMC meeting
    out["fomc_window"] = 0
    for fd in fomc:
        mask = (out[date_col] >= fd - pd.Timedelta(days=7)) & (
            out[date_col] <= fd + pd.Timedelta(days=7)
        )
        out.loc[mask, "fomc_window"] = 1

    print(
        f"[features] FOMC windows flagged (±7 days): "
        f"{out['fomc_window'].sum()} / {len(out)} days."
    )
    return out


# ════════════════════════════════════════════════════════════
# SECTION D:  MASTER FEATURE TABLE
# ════════════════════════════════════════════════════════════

def build_feature_table(options_df, spx_df, rf_series=None):
    """
    Orchestrate all feature computations and merge into a single
    date-indexed DataFrame.

    Parameters
    ----------
    options_df : pd.DataFrame
        Cleaned options data.
    spx_df : pd.DataFrame
        Daily SPX price series.
    rf_series : pd.Series, optional
        1-month risk-free rate.

    Returns
    -------
    pd.DataFrame
        Master feature table indexed by date, containing:
        - spx_close, log_return
        - rv_daily, rv_weekly, rv_monthly, rvol_*
        - bv_monthly
        - fwd_rv, fwd_rvol  (forward-looking labels)
        - atm_iv_30d
        - mfiv_30d, mfiv_vol_30d
        - fomc_window
    """
    print("\n" + "=" * 60)
    print("  BUILDING MASTER FEATURE TABLE")
    print("=" * 60)

    # ── Realised volatility ────────────────────────────────
    rv = compute_realized_variance(spx_df)
    bv = compute_bipower_variation(spx_df)
    fwd = compute_forward_rv(spx_df)

    # ── Implied volatility ─────────────────────────────────
    atm = compute_atm_iv(options_df)
    mfiv_raw = compute_model_free_iv(options_df, rf_series=rf_series)
    mfiv = compute_mfiv_30d(mfiv_raw) if len(mfiv_raw) > 0 else pd.DataFrame()

    # ── Merge everything on date ───────────────────────────
    master = rv.copy()
    master = master.merge(bv, on="date", how="left")
    master = master.merge(fwd, on="date", how="left")
    master = master.merge(atm, on="date", how="left")
    if len(mfiv) > 0:
        master = master.merge(mfiv, on="date", how="left")

    # ── Event flags ────────────────────────────────────────
    master = add_event_flags(master)

    master = master.sort_values("date").reset_index(drop=True)
    print(f"\n[features] Master table: {len(master)} rows, {master.shape[1]} columns.")
    print(f"[features] Date range: {master['date'].min().date()} → "
          f"{master['date'].max().date()}")
    print(f"[features] Columns: {list(master.columns)}")

    return master
