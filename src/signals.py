# ============================================================
# signals.py — Trading Signal Construction
# ============================================================
"""
Builds trading signals:

    1. **Level Signal** — Variance Risk Premium (VRP)
       S = IV − forecast_RV ;  z-score normalised

    2. **Surface Relative-Value** — Skew & term-structure mispricing
       Compare implied vs physical downside probabilities

All signals output a standardised z-score ∈ ℝ on each trading date.
"""

import numpy as np
import pandas as pd

from src.config import (
    SIGNAL_ZSCORE_ENTRY,
    SIGNAL_LOOKBACK,
)


# ════════════════════════════════════════════════════════════
# 1.  VARIANCE RISK PREMIUM (VRP) — Level Signal
# ════════════════════════════════════════════════════════════

def compute_vrp_signal(feature_df, iv_col="atm_iv_at_expiry", rv_col=None):
    """
    Core signal: VRP = IV − forecast RV

    The z-scored version:
        z = (VRP − rolling_mean) / rolling_std

    Trading rule:
        z > +threshold  →  sell vol (IV is rich)
        z < −threshold  →  buy vol  (IV is cheap)

    Parameters
    ----------
    feature_df : pd.DataFrame
        Must contain an IV column and at least one RV forecast column.
    iv_col : str
        Column name for implied volatility.
    rv_col : str, optional
        Column name for RV forecast.  If None, tries 'composite_rv_forecast',
        then falls back to 'har_rv_forecast', then 'garch_forecast'.

    Returns
    -------
    pd.DataFrame
        Columns: date, iv, rv_forecast, vrp, vrp_zscore, signal
    """
    df = feature_df.copy()

    # ── Select RV forecast column (must be model forecast, never future data) ─
    FORBIDDEN_RV_COLS = {"fwd_rv", "fwd_rvol"}  # future realized vol = label only, would leak
    if rv_col is None:
        for candidate in [
            "composite_rv_forecast",
            "har_rv_forecast",
            "garch_forecast",
        ]:
            if candidate in df.columns and candidate not in FORBIDDEN_RV_COLS and df[candidate].notna().sum() > 50:
                rv_col = candidate
                break
    if rv_col in FORBIDDEN_RV_COLS:
        raise ValueError("rv_col must not be fwd_rv/fwd_rvol (future data); use forecast columns only.")
    if rv_col is None:
        raise ValueError("No valid RV forecast column found in feature_df.")

    # ── Convert variance forecasts to volatility if needed ─
    # If the IV column is in vol units (~0.05–0.80) and rv_col is in
    # variance units (~0.0025–0.64), convert rv to vol first.
    rv_vals = df[rv_col].dropna()
    iv_vals = df[iv_col].dropna()

    # Heuristic: if median RV >> median IV², RV is in variance terms
    if rv_vals.median() > 2 * (iv_vals.median() ** 2):
        rv_numeric = pd.to_numeric(df[rv_col], errors="coerce").clip(lower=0)
        df["rv_forecast_vol"] = np.sqrt(rv_numeric.astype(float))
        rv_use = "rv_forecast_vol"
    else:
        rv_use = rv_col

    # ── Compute VRP ────────────────────────────────────────
    df["vrp"] = df[iv_col] - df[rv_use]

    # ── Z-score normalisation ──────────────────────────────
    df["vrp_mean"] = df["vrp"].rolling(SIGNAL_LOOKBACK, min_periods=60).mean()
    df["vrp_std"] = df["vrp"].rolling(SIGNAL_LOOKBACK, min_periods=60).std()
    df["vrp_zscore"] = (df["vrp"] - df["vrp_mean"]) / df["vrp_std"].replace(0, np.nan)

    # ── Discrete signal: +1 (short vol), −1 (long vol), 0 (flat)
    df["signal"] = 0
    df.loc[df["vrp_zscore"] > SIGNAL_ZSCORE_ENTRY, "signal"] = 1    # short vol
    df.loc[df["vrp_zscore"] < -SIGNAL_ZSCORE_ENTRY, "signal"] = -1  # long vol

    base_cols = ["date", iv_col, rv_use, "vrp", "vrp_zscore", "signal"]
    expiry_cols = [c for c in ["exdate_trade", "dte_trade"] if c in df.columns]
    out = df[base_cols + expiry_cols].copy()
    out = out.rename(columns={iv_col: "iv", rv_use: "rv_forecast"})
    out = out.dropna(subset=["vrp_zscore"])

    n_short = (out["signal"] == 1).sum()
    n_long = (out["signal"] == -1).sum()
    n_flat = (out["signal"] == 0).sum()
    print(f"[signals] VRP signal: {len(out)} days — "
          f"short_vol={n_short}, long_vol={n_long}, flat={n_flat}")

    return out


# ════════════════════════════════════════════════════════════
# 2.  SURFACE RELATIVE-VALUE SIGNAL
# ════════════════════════════════════════════════════════════

def compute_skew_signal(options_df, spx_df):
    """
    Skew mispricing: compare implied vs realised downside frequency.

    For each date:
        - Implied P(down 5%) from OTM put IV via Black-Scholes Δ
        - Historical P(down 5%) from trailing 252-day returns

    Signal = implied_tail − realised_tail  (positive → puts are rich)

    Parameters
    ----------
    options_df : pd.DataFrame
        Cleaned options data.
    spx_df : pd.DataFrame
        Daily SPX returns.

    Returns
    -------
    pd.DataFrame
        Columns: date, implied_tail_prob, realised_tail_prob, skew_signal
    """
    # ── Realised tail frequency ────────────────────────────
    ret = spx_df.set_index("date")["log_return"]
    tail_threshold = -0.02  # 2% daily drop

    realised_tail = (
        (ret < tail_threshold)
        .astype(float)
        .rolling(252, min_periods=60)
        .mean()
    )
    realised_tail.name = "realised_tail_prob"

    # ── Implied tail probability from OTM puts ─────────────
    # Use puts with moneyness ≈ 0.95 (5% OTM)
    puts = options_df[options_df["cp_flag"] == "P"].copy()
    puts = puts[(puts["moneyness"] >= 0.93) & (puts["moneyness"] <= 0.97)]

    # Group by date, take median |delta| as implied tail probability
    implied_tail = (
        puts.groupby("date")["delta"]
        .apply(lambda x: x.abs().median())
        .rename("implied_tail_prob")
    )

    # ── Merge and compute signal ───────────────────────────
    combo = pd.DataFrame({"implied_tail_prob": implied_tail,
                          "realised_tail_prob": realised_tail}).dropna()
    combo["skew_signal"] = combo["implied_tail_prob"] - combo["realised_tail_prob"]
    combo = combo.reset_index().rename(columns={"index": "date"})

    print(f"[signals] Skew signal: {len(combo)} dates computed.")
    return combo


def compute_term_structure_signal(options_df):
    """
    Term-structure mispricing: short-dated IV vs long-dated IV.

    signal = IV_short − IV_long
    Positive → short-end is rich relative to long-end (contango)

    Parameters
    ----------
    options_df : pd.DataFrame
        Cleaned options data with dte and impl_volatility.

    Returns
    -------
    pd.DataFrame
        Columns: date, iv_short, iv_long, term_signal
    """
    df = options_df.copy()
    # Restrict to near-ATM
    df = df[(df["moneyness"] >= 0.97) & (df["moneyness"] <= 1.03)]

    # Short-dated: 7–14 DTE; Long-dated: 30–60 DTE
    short = df[(df["dte"] >= 7) & (df["dte"] <= 14)]
    long_ = df[(df["dte"] >= 30) & (df["dte"] <= 60)]

    iv_short = short.groupby("date")["impl_volatility"].median().rename("iv_short")
    iv_long = long_.groupby("date")["impl_volatility"].median().rename("iv_long")

    combo = pd.DataFrame({"iv_short": iv_short, "iv_long": iv_long}).dropna()
    combo["term_signal"] = combo["iv_short"] - combo["iv_long"]
    combo = combo.reset_index().rename(columns={"index": "date"})

    print(f"[signals] Term-structure signal: {len(combo)} dates computed.")
    return combo


# ════════════════════════════════════════════════════════════
# 3.  MASTER SIGNAL TABLE
# ════════════════════════════════════════════════════════════

def build_signal_table(feature_df, options_df, spx_df):
    """
    Compute all signals and merge into one DataFrame.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Master feature table (with RV forecasts merged).
    options_df : pd.DataFrame
        Cleaned options data.
    spx_df : pd.DataFrame
        Daily SPX prices / returns.

    Returns
    -------
    pd.DataFrame
        Date-indexed table with all signals.
    """
    print("\n" + "=" * 60)
    print("  BUILDING SIGNAL TABLE")
    print("=" * 60)

    # ── VRP (primary signal) ───────────────────────────────
    vrp = compute_vrp_signal(feature_df)

    # ── Surface signals ────────────────────────────────────
    skew = compute_skew_signal(options_df, spx_df)
    term = compute_term_structure_signal(options_df)

    # ── Merge ──────────────────────────────────────────────
    signals = vrp.copy()
    signals = signals.merge(skew, on="date", how="left")
    signals = signals.merge(term, on="date", how="left")

    signals = signals.sort_values("date").reset_index(drop=True)
    print(f"\n[signals] Signal table: {len(signals)} rows, "
          f"{signals.shape[1]} columns.")

    return signals
