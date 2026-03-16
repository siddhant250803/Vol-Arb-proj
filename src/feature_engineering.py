"""
IV measures (ATM, model-free), realized variance, bipower variation, event flags.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.config import (
    COL,
    ATM_DELTA_BAND,
    VARIANCE_SWAP_STRIKE_BAND,
    TRADING_DAYS_PER_YEAR,
    RV_HORIZONS,
    ANNUALISATION_FACTOR,
    FOMC_DATES,
)


def compute_atm_iv_at_expiry(options_df, expiry_df):
    """
    For each trading date, extract ATM IV at the time-to-expiry of the
    tradeable option (dte_trade). Matches IV to the actual holding period.

    Parameters
    ----------
    options_df : pd.DataFrame
        Cleaned options with columns: date, dte, delta, impl_volatility, cp_flag.
    expiry_df : pd.DataFrame
        Columns: date, exdate_trade, dte_trade (from compute_tradeable_expiry).

    Returns
    -------
    pd.DataFrame
        Columns: date, atm_iv_at_expiry
    """
    if expiry_df.empty:
        return pd.DataFrame(columns=["date", "atm_iv_at_expiry"])

    df = options_df.copy()
    df["abs_delta"] = df[COL["delta"]].abs()
    atm_mask = (df["abs_delta"] >= 0.50 - ATM_DELTA_BAND) & (
        df["abs_delta"] <= 0.50 + ATM_DELTA_BAND
    )
    atm = df.loc[atm_mask].copy()
    atm["w"] = 1.0 / (1e-6 + (atm["abs_delta"] - 0.50).abs())

    def _wavg(g):
        return np.average(g[COL["iv"]], weights=g["w"])

    grouped_raw = atm.groupby([COL["date"], "dte"]).apply(
        _wavg, include_groups=False,
    )
    grouped = grouped_raw.reset_index()
    grouped.columns = ["date", "dte", "atm_iv"]

    results = []
    for _, exp_row in expiry_df.iterrows():
        date, dte_trade = exp_row["date"], exp_row["dte_trade"]
        if pd.isna(dte_trade):
            continue
        dte_trade = int(dte_trade)
        grp = grouped[grouped["date"] == date]
        if grp.empty:
            continue
        grp = grp.sort_values("dte")
        dtes = grp["dte"].values
        ivs = grp["atm_iv"].values
        if len(dtes) == 1:
            iv_val = float(ivs[0])
        else:
            try:
                f = interp1d(
                    dtes.astype(float),
                    ivs.astype(float),
                    kind="linear",
                    fill_value="extrapolate",
                )
                iv_val = float(f(dte_trade))
            except Exception:
                continue
        if 0.01 < iv_val < 2.0:
            results.append({"date": date, "atm_iv_at_expiry": iv_val})

    out = pd.DataFrame(results)
    if len(out) > 0:
        out["date"] = pd.to_datetime(out["date"])
    print(f"[features] ATM IV (at expiry): {len(out)} dates computed.")
    return out


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

        r = 0.0
        if rf_series is not None and date in rf_series.index:
            r = rf_series.loc[date]

        if COL["forward"] in grp.columns and grp[COL["forward"]].notna().any():
            F = grp[COL["forward"]].dropna().iloc[0]
        else:
            F = spot * np.exp(r * T)

        lo = F * (1.0 - VARIANCE_SWAP_STRIKE_BAND)
        hi = F * (1.0 + VARIANCE_SWAP_STRIKE_BAND)
        sub = grp[(grp["strike"] >= lo) & (grp["strike"] <= hi)].copy()
        if len(sub) < 4:
            continue

        puts = sub[(sub[COL["cp_flag"]] == "P") & (sub["strike"] < F)]
        calls = sub[(sub[COL["cp_flag"]] == "C") & (sub["strike"] >= F)]
        otm = pd.concat([puts, calls]).sort_values("strike")

        if len(otm) < 3:
            continue

        strikes = otm["strike"].values
        prices = otm["mid"].values

        variance = 0.0
        for i in range(len(strikes)):
            K = strikes[i]
            Q = prices[i]
            if i == 0:
                dK = strikes[1] - strikes[0]
            elif i == len(strikes) - 1:
                dK = strikes[-1] - strikes[-2]
            else:
                dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

            variance += (dK / (K ** 2)) * np.exp(r * T) * Q

        mfiv = (2.0 / T) * variance

        if mfiv > 0:
            results.append({
                "date": date,
                "dte": dte,
                "mfiv": mfiv,
                "mfiv_vol": np.sqrt(mfiv),
            })

    out = pd.DataFrame(results)
    if len(out) > 0:
        out["date"] = pd.to_datetime(out["date"])
    print(f"[features] Model-free IV: {len(out)} (date, dte) pairs computed.")
    return out


def compute_mfiv_at_expiry(mfiv_df, expiry_df):
    """
    Interpolate model-free implied variance to the tradeable expiry (dte_trade)
    for each date. Matches MFIV to the actual holding period.

    Parameters
    ----------
    mfiv_df : pd.DataFrame
        Output of ``compute_model_free_iv()`` (date, dte, mfiv, mfiv_vol).
    expiry_df : pd.DataFrame
        Columns: date, exdate_trade, dte_trade.

    Returns
    -------
    pd.DataFrame
        Columns: date, mfiv_at_expiry, mfiv_vol_at_expiry
    """
    if mfiv_df.empty or expiry_df.empty:
        return pd.DataFrame(columns=["date", "mfiv_at_expiry", "mfiv_vol_at_expiry"])

    results = []
    for _, exp_row in expiry_df.iterrows():
        date, dte_trade = exp_row["date"], exp_row["dte_trade"]
        if pd.isna(dte_trade):
            continue
        dte_trade = int(dte_trade)
        grp = mfiv_df[mfiv_df["date"] == date].sort_values("dte")
        if len(grp) < 2:
            if len(grp) == 1:
                var_val = grp["mfiv"].iloc[0]
            else:
                continue
        else:
            try:
                f_var = interp1d(
                    grp["dte"].values.astype(float),
                    grp["mfiv"].values.astype(float),
                    kind="linear",
                    fill_value="extrapolate",
                )
                var_val = float(f_var(dte_trade))
            except Exception:
                continue
        if var_val > 0:
            results.append({
                "date": date,
                "mfiv_at_expiry": var_val,
                "mfiv_vol_at_expiry": np.sqrt(var_val),
            })

    out = pd.DataFrame(results)
    if len(out) > 0:
        out["date"] = pd.to_datetime(out["date"])
    print(f"[features] MFIV (at expiry): {len(out)} dates interpolated.")
    return out


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

    for label in windows:
        df[f"rvol_{label}"] = np.sqrt(df[f"rv_{label}"])

    rv_cols = [f"rv_{k}" for k in windows.keys()] + [f"rvol_{k}" for k in windows.keys()]
    df = df.drop(columns=["return_sq"]).dropna(subset=rv_cols)
    print(f"[features] Realized variance: {len(df)} rows for {list(windows.keys())}.")
    return df


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


def compute_forward_rv_at_expiry(spx_df, expiry_df):
    """
    Compute the *future* realised variance from entry to expiry for each date.

    Uses the actual expiry date (exdate_trade) to count the real number of
    trading days between entry and expiry, avoiding the calendar-DTE vs
    trading-day mismatch (e.g. 7 calendar days Fri→Fri = 5 trading days).

    Parameters
    ----------
    spx_df : pd.DataFrame
        Must have columns: date, log_return.
    expiry_df : pd.DataFrame
        Columns: date, exdate_trade, dte_trade.

    Returns
    -------
    pd.DataFrame
        Columns: date, fwd_rv, fwd_rvol (annualised), fwd_trading_days
    """
    if expiry_df.empty:
        return pd.DataFrame(columns=["date", "fwd_rv", "fwd_rvol", "fwd_trading_days"])

    df = spx_df[["date", "log_return"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return_sq = df["log_return"].values ** 2
    date_to_idx = {d: i for i, d in enumerate(df["date"])}

    results = []
    for _, exp_row in expiry_df.iterrows():
        entry = pd.to_datetime(exp_row["date"])
        exdate = pd.to_datetime(exp_row["exdate_trade"])
        if pd.isna(exdate) or entry not in date_to_idx:
            continue

        entry_idx = date_to_idx[entry]

        # Find actual trading-day index of expiry (or nearest prior trading day)
        if exdate in date_to_idx:
            expiry_idx = date_to_idx[exdate]
        else:
            candidates = [d for d in date_to_idx if d <= exdate and d > entry]
            if not candidates:
                continue
            expiry_idx = date_to_idx[max(candidates)]

        n_trading_days = expiry_idx - entry_idx
        if n_trading_days <= 0 or entry_idx + n_trading_days >= len(df):
            continue

        fwd_sum = return_sq[entry_idx + 1: entry_idx + 1 + n_trading_days].sum()
        fwd_rv = fwd_sum * (ANNUALISATION_FACTOR / n_trading_days)
        results.append({
            "date": entry,
            "fwd_rv": fwd_rv,
            "fwd_rvol": np.sqrt(max(fwd_rv, 0)),
            "fwd_trading_days": n_trading_days,
        })

    out = pd.DataFrame(results)
    if len(out) > 0:
        out["date"] = pd.to_datetime(out["date"])
    print(f"[features] Forward RV (at expiry): {len(out)} rows.")
    return out


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
        fomc_window (1 if within ±7 days of FOMC date)
    """
    fomc = pd.to_datetime(FOMC_DATES)
    out = df.copy()

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


def compute_tradeable_expiry(options_df):
    """
    For each date, get the exdate of the shortest-dated option in range
    (the one we would trade). Options expire every Friday.

    Parameters
    ----------
    options_df : pd.DataFrame
        Cleaned options with date, exdate, dte.

    Returns
    -------
    pd.DataFrame
        Columns: date, exdate_trade, dte_trade
    """
    if options_df.empty or "exdate" not in options_df.columns:
        return pd.DataFrame(columns=["date", "exdate_trade", "dte_trade"])
    idx = options_df.groupby(COL["date"])["dte"].idxmin()
    out = options_df.loc[idx, [COL["date"], COL["exdate"], "dte"]].drop_duplicates()
    out = out.rename(columns={COL["exdate"]: "exdate_trade", "dte": "dte_trade"})
    out = out[[COL["date"], "exdate_trade", "dte_trade"]].reset_index(drop=True)
    print(f"[features] Tradeable expiry: {len(out)} dates with exdate_trade.")
    return out


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
        - fwd_rv, fwd_rvol  (forward RV at expiry horizon)
        - atm_iv_at_expiry
        - mfiv_at_expiry, mfiv_vol_at_expiry
        - fomc_window
    """
    expiry_df = compute_tradeable_expiry(options_df)

    rv = compute_realized_variance(spx_df)
    bv = compute_bipower_variation(spx_df)
    fwd = compute_forward_rv_at_expiry(spx_df, expiry_df)

    atm = compute_atm_iv_at_expiry(options_df, expiry_df)
    mfiv_raw = compute_model_free_iv(options_df, rf_series=rf_series)
    mfiv = compute_mfiv_at_expiry(mfiv_raw, expiry_df) if len(mfiv_raw) > 0 else pd.DataFrame()

    master = rv.copy()
    master = master.merge(bv, on="date", how="left")
    master = master.merge(fwd, on="date", how="left")
    master = master.merge(atm, on="date", how="left")
    if len(mfiv) > 0:
        master = master.merge(mfiv, on="date", how="left")
    if len(expiry_df) > 0:
        master = master.merge(expiry_df, on="date", how="left")

    master = add_event_flags(master)

    master = master.sort_values("date").reset_index(drop=True)
    print(f"\n[features] Master table: {len(master)} rows, {master.shape[1]} columns.")
    print(f"[features] Date range: {master['date'].min().date()} → "
          f"{master['date'].max().date()}")
    print(f"[features] Columns: {list(master.columns)}")

    return master
