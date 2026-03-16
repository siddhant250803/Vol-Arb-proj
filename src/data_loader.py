"""
Data loading, cleaning, and preparation for OptionMetrics SPX options.
Downloads SPX prices from Yahoo Finance, applies quality filters, computes derived columns.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

from src.config import (
    OPTIONS_FILE,
    OPTIONS_USECOLS,
    YIELD_FILE,
    COL,
    STRIKE_DIVISOR,
    MIN_BID,
    MAX_SPREAD_RATIO,
    MIN_DTE,
    MAX_DTE,
    MONEYNESS_BAND,
    TRADING_DAYS_PER_YEAR,
)


def load_options_raw(filepath=None, nrows=None):
    """
    Read the OptionMetrics CSV into a DataFrame with correct dtypes.

    Handles both the *filtered* file (has ``spx_price``) and the *full*
    daily file (no ``spx_price``; spot is derived later from
    ``forward_price``).

    Parameters
    ----------
    filepath : Path or str, optional
        Defaults to ``config.OPTIONS_FILE`` (the full daily dataset).
    nrows : int, optional
        Read only the first *nrows* rows (useful for development).

    Returns
    -------
    pd.DataFrame
        Raw options data with date columns parsed.
    """
    filepath = filepath or OPTIONS_FILE
    if not Path(filepath).exists():
        gz_path = Path(f"{filepath}.gz")
        if gz_path.exists():
            filepath = gz_path
    print(f"[data_loader] Loading options from {filepath.name} ...")

    # Only load the columns we need (saves memory on 5M+ row files)
    usecols = None
    try:
        header = pd.read_csv(filepath, nrows=0).columns.tolist()
        available = [c for c in OPTIONS_USECOLS if c in header]
        if len(available) >= len(OPTIONS_USECOLS) * 0.8:
            usecols = available
    except Exception:
        pass

    df = pd.read_csv(
        filepath,
        usecols=usecols,
        parse_dates=["date", "exdate"],
        low_memory=False,
        nrows=nrows,
    )
    print(f"[data_loader]   → {len(df):,} rows loaded.")
    required = ["date", "exdate", "impl_volatility", "strike_price", "best_bid", "best_offer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[data_loader] Options CSV missing required columns: {missing}. Check file or usecols.")
    return df


def _derive_spot_price(df):
    """
    When the raw data lacks ``spx_price``, derive a daily SPX spot
    estimate from the options data.

    Strategy (in order of preference):
      A. If ``forward_price`` is populated → use it (F ≈ S for short T).
      B. Otherwise, use the **strike of the most ATM option**:
         for options with |delta| closest to 0.50, K ≈ S.

    For each date we take the median strike of the options whose
    |delta| is closest to 0.50 on the shortest available expiry.

    Parameters
    ----------
    df : pd.DataFrame
        Raw options data with columns: date, exdate, delta, strike_price,
        and optionally forward_price.

    Returns
    -------
    pd.DataFrame
        Same data with a new ``spx_price`` column added.
    """
    out = df.copy()

    has_fwd = (
        "forward_price" in out.columns
        and out["forward_price"].notna().sum() > 100
    )
    if has_fwd:
        out["_dte"] = (out["exdate"] - out["date"]).dt.days
        out["_abs_delta"] = out["delta"].abs()
        min_dte = out.groupby("date")["_dte"].transform("min")
        shortest = out[out["_dte"] == min_dte]
        atm_mask = (shortest["_abs_delta"] >= 0.35) & (shortest["_abs_delta"] <= 0.65)
        atm = shortest.loc[atm_mask]
        spot_map = atm.groupby("date")["forward_price"].median().rename("spx_price")
        all_fwd = out.groupby("date")["forward_price"].median().rename("spx_price")
        spot_map = spot_map.reindex(all_fwd.index).fillna(all_fwd)
        out = out.drop(columns=["_dte", "_abs_delta"], errors="ignore")
        out = out.merge(spot_map, on="date", how="left")
        n_dates = spot_map.notna().sum()
        print(f"[data_loader] Derived SPX spot from forward_price for {n_dates} dates.")
        return out

    out["_strike_real"] = out[COL["strike_raw"]] / STRIKE_DIVISOR
    out["_abs_delta"] = out["delta"].abs()

    has_delta = out["_abs_delta"].notna()
    valid = out.loc[has_delta].copy()
    valid["_delta_dist"] = (valid["_abs_delta"] - 0.50).abs()

    atm = (
        valid.sort_values(["date", "_delta_dist"])
        .groupby("date")
        .head(5)
    )

    spot_map = (
        atm.groupby("date")["_strike_real"]
        .median()
        .rename("spx_price")
    )

    out = out.drop(columns=["_strike_real", "_abs_delta"], errors="ignore")
    out = out.merge(spot_map, on="date", how="left")

    n_dates = spot_map.notna().sum()
    print(f"[data_loader] Derived SPX spot from ATM strikes for {n_dates} dates.")
    return out


def clean_options(df, min_dte=None, max_dte=None):
    """
    Apply quality filters and compute derived columns.

    Steps
    -----
    1. If ``spx_price`` is missing, derive it from ``forward_price``.
    2. Convert strike from ×1000 to actual dollar value.
    3. Compute mid-price, bid-ask spread ratio, moneyness, DTE.
    4. Drop rows with missing IV or negative bids.
    5. Apply configurable filters (min bid, spread, DTE, moneyness).

    Parameters
    ----------
    df : pd.DataFrame
        Raw options data from ``load_options_raw``.
    min_dte : int, optional
        Override config MIN_DTE.
    max_dte : int, optional
        Override config MAX_DTE.

    Returns
    -------
    pd.DataFrame
        Cleaned options data with new columns:
        strike, mid, spread, spread_ratio, dte, moneyness, log_moneyness
    """
    dte_lo = min_dte if min_dte is not None else MIN_DTE
    dte_hi = max_dte if max_dte is not None else MAX_DTE

    out = df.copy()
    n_start = len(out)

    if COL["spot"] not in out.columns:
        out = _derive_spot_price(out)

    # Drop rows where spot could not be determined
    out = out.dropna(subset=[COL["spot"]])

    out["strike"] = out[COL["strike_raw"]] / STRIKE_DIVISOR
    out["mid"] = (out[COL["bid"]] + out[COL["ask"]]) / 2.0
    out["spread"] = out[COL["ask"]] - out[COL["bid"]]
    out["spread_ratio"] = out["spread"] / out["mid"].replace(0, np.nan)
    out["dte"] = (out[COL["exdate"]] - out[COL["date"]]).dt.days
    out["moneyness"] = out["strike"] / out[COL["spot"]]
    out["log_moneyness"] = np.log(out["moneyness"])

    out = out.dropna(subset=[COL["iv"]])
    out = out[out[COL["iv"]] > 0]
    out = out[out[COL["bid"]] >= MIN_BID]
    out = out[out["spread_ratio"] <= MAX_SPREAD_RATIO]
    out = out[(out["dte"] >= dte_lo) & (out["dte"] <= dte_hi)]
    out = out[
        (out["moneyness"] >= 1.0 - MONEYNESS_BAND)
        & (out["moneyness"] <= 1.0 + MONEYNESS_BAND)
    ]

    n_end = len(out)
    print(
        f"[data_loader] Cleaned options (DTE {dte_lo}–{dte_hi}): {n_start:,} → {n_end:,} "
        f"({n_start - n_end:,} rows removed)"
    )
    return out.reset_index(drop=True)


def extract_spx_prices(df):
    """
    Download daily SPX closing prices from Yahoo Finance (^GSPC). Requires
    network access. Covers the options date range with a 1-year buffer on
    each side for rolling-window model burn-in.

    Parameters
    ----------
    df : pd.DataFrame
        Options data (used only to determine the date range).

    Returns
    -------
    pd.DataFrame
        Columns: date, spx_close, log_return, abs_return
        Sorted by date, one row per trading day.
    """
    date_col = COL["date"] if COL["date"] in df.columns else "date"
    dates = pd.to_datetime(df[date_col])
    start = (dates.min() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    print(f"[data_loader] Downloading SPX (^GSPC) from Yahoo Finance ({start} to {end}) ...")
    ticker = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(ticker.columns, pd.MultiIndex):
        ticker.columns = ticker.columns.get_level_values(0)

    prices = pd.DataFrame({
        "date": ticker.index,
        "spx_close": ticker["Close"].values,
    }).reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices = prices.dropna(subset=["spx_close"]).sort_values("date").reset_index(drop=True)

    prices["log_return"] = np.log(prices["spx_close"] / prices["spx_close"].shift(1))
    prices["abs_return"] = prices["log_return"].abs()
    prices = prices.dropna(subset=["log_return"]).reset_index(drop=True)

    print(
        f"[data_loader] SPX price series: {len(prices)} trading days "
        f"({prices['date'].min().date()} to {prices['date'].max().date()})"
    )
    return prices


def load_yield_curve(filepath=None):
    """
    Load the daily zero-coupon yield panel.

    The file has columns [date, MAX_DATA_TTM, 1, 2, …, 360]
    where integer columns are *monthly* maturities.

    Returns
    -------
    pd.DataFrame
        Index = date, columns = maturity in *months* (int).
        Values are continuously compounded annual yields.
    """
    filepath = filepath or YIELD_FILE
    print(f"[data_loader] Loading yield curve from {filepath.name} ...")

    raw = pd.read_csv(filepath, low_memory=False)

    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.rename(columns={date_col: "date"})

    if "MAX_DATA_TTM" in raw.columns:
        raw = raw.drop(columns=["MAX_DATA_TTM"])

    raw = raw.set_index("date").sort_index()

    def _safe_int(c):
        try:
            return int(float(c))
        except (ValueError, TypeError):
            return c
    raw.columns = [_safe_int(c) for c in raw.columns]

    print(
        f"[data_loader]   → {len(raw):,} days, maturities 1–{raw.columns.max()} months"
    )
    return raw


def get_risk_free_rate(yield_df, target_months=1):
    """
    Extract a single maturity series (annualised, continuously compounded).

    Parameters
    ----------
    yield_df : pd.DataFrame
        Output of ``load_yield_curve()``.
    target_months : int
        Which maturity column to extract (default: 1-month).

    Returns
    -------
    pd.Series
        Indexed by date, values = annualised yield.
    """
    if target_months not in yield_df.columns:
        raise KeyError(f"Maturity {target_months} not in yield panel.")
    series = yield_df[target_months].dropna()
    series.name = "risk_free_rate"
    return series


def load_all_data(nrows=None):
    """
    One-call loader that returns everything needed by downstream modules.

    Returns
    -------
    dict with keys:
        "options"      – entry-day options (DTE 5–9), used for VRP signal
        "options_wide" – wider DTE options (DTE 5–60), used for term-structure analysis
        "spx"          – daily SPX price / return series
        "yields"       – full yield panel
        "rf"           – 1-month risk-free rate series
    """
    raw = load_options_raw(nrows=nrows)
    spx = extract_spx_prices(raw)

    # Merge yfinance SPX closes into raw options so clean_options
    # uses real closing prices for moneyness instead of deriving from strikes.
    spx_map = spx[["date", "spx_close"]].rename(columns={"spx_close": COL["spot"]})
    raw = raw.drop(columns=[COL["spot"]], errors="ignore")
    raw = raw.merge(spx_map, on="date", how="left")

    options = clean_options(raw)
    options_wide = clean_options(raw, min_dte=5, max_dte=60)
    yields = load_yield_curve()
    rf = get_risk_free_rate(yields, target_months=1)

    return {
        "options": options,
        "options_wide": options_wide,
        "spx": spx,
        "yields": yields,
        "rf": rf,
    }
