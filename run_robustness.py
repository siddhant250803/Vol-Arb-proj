#!/usr/bin/env python3
# ============================================================
# run_robustness.py — Robustness, Out-of-Sample & Stress Tests
# ============================================================
"""
Comprehensive robustness battery for the VRP strategy:

    A. Out-of-sample test (train 60 % / test 40 %)
    B. Walk-forward analysis (5 non-overlapping windows)
    C. Yearly performance breakdown
    D. Stress-period analysis (GFC, Volmageddon, COVID, …)
    E. Volatility-regime conditioning (high-vol vs low-vol)
    F. Parameter sensitivity grid (threshold × hold-days × cost)
    G. Bootstrap Sharpe confidence intervals

All backtests hold only until option expiry (Friday weeklies). Hold-days sensitivity uses max 5d (≤ expiry).

Produces:
    output/figures/15_oos_walkforward.png
    output/figures/16_stress_periods.png
    output/figures/17_regime_analysis.png
    output/figures/18_param_sensitivity.png
    output/figures/19_bootstrap_sharpe.png
    output/figures/20_yearly_breakdown.png
    output/reports/robustness_report.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR,
    SIGNAL_ZSCORE_ENTRY, SIGNAL_LOOKBACK,
    NOTIONAL_CAPITAL, TRADING_DAYS_PER_YEAR,
    POSITION_HOLD_DAYS, TRANSACTION_COST_BPS,
    PLOT_COLORS, PLOT_PALETTE, PLOT_PRIMARY, PLOT_SECONDARY,
    PLOT_ACCENT, PLOT_NEUTRAL, PLOT_POSITIVE, PLOT_NEGATIVE,
)
from src.data_loader import load_all_data
from src.feature_engineering import build_feature_table
from src.rv_models import run_all_rv_models
from src.signals import compute_vrp_signal
from src.backtest import run_backtest, trades_to_dataframe
from src.performance import (
    sharpe_ratio, sortino_ratio, annualised_return,
    annualised_volatility, compute_drawdown, calmar_ratio,
    return_statistics, benchmark_returns_from_spx, benchmark_comparison,
    probabilistic_sharpe_ratio,
    returns_on_full_calendar,
)

warnings.filterwarnings("ignore")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-v0_8")

REPORT = []   # accumulate report lines


def _log(msg):
    print(msg)
    REPORT.append(msg)


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [viz] Saved → {path.relative_to(FIGURES_DIR.parent.parent)}")
    plt.close(fig)


def _metrics(dr):
    """Return a dict of standard metrics from a daily-return Series (incl. PSR)."""
    if len(dr) < 10:
        return {k: np.nan for k in [
            "ann_ret", "ann_vol", "sharpe", "sortino",
            "calmar", "max_dd", "skew", "kurt", "n_days", "psr",
        ]}
    dd = compute_drawdown(dr)
    cal = annualised_return(dr) / abs(dd["max_drawdown"]) if dd["max_drawdown"] != 0 else np.inf
    return {
        "ann_ret": annualised_return(dr),
        "ann_vol": annualised_volatility(dr),
        "sharpe": sharpe_ratio(dr),
        "sortino": sortino_ratio(dr),
        "calmar": cal,
        "max_dd": dd["max_drawdown"],
        "skew": dr.skew(),
        "kurt": dr.kurtosis(),
        "n_days": len(dr),
        "psr": probabilistic_sharpe_ratio(dr, sr_ref=0.0),
    }


def _dr_for_metrics(pnl_df, spx_df):
    """Calendar-aligned daily returns for metrics (flat days = 0)."""
    if pnl_df.empty:
        return pd.Series(dtype=float)
    if spx_df is not None and not spx_df.empty:
        dr = returns_on_full_calendar(pnl_df, spx_df)
        if dr is not None and len(dr) >= 2:
            return dr.dropna()
    return pnl_df["daily_return"].dropna()


# ════════════════════════════════════════════════════════════
#  A.  OUT-OF-SAMPLE TEST
# ════════════════════════════════════════════════════════════

def oos_test(signal_df, spx_df, train_frac=0.60):
    """Split signal chronologically: train first 60 %, test last 40 %."""
    _log("\n" + "=" * 65)
    _log("  A. OUT-OF-SAMPLE TEST (Train 60 % / Test 40 %)")
    _log("=" * 65)

    dates = sorted(signal_df["date"].unique())
    split_idx = int(len(dates) * train_frac)
    split_date = dates[split_idx]

    train_sig = signal_df[signal_df["date"] < split_date]
    test_sig = signal_df[signal_df["date"] >= split_date]

    results = {}
    for label, sig in [("IN-SAMPLE", train_sig), ("OUT-OF-SAMPLE", test_sig)]:
        trades, pnl = run_backtest(sig, spx_df)
        if pnl.empty:
            results[label] = {"n_trades": 0}
            continue
        m = _metrics(_dr_for_metrics(pnl, spx_df))
        m["n_trades"] = len(trades)
        m["total_pnl"] = sum(t.net_pnl for t in trades)
        m["win_rate"] = sum(1 for t in trades if t.net_pnl > 0) / max(len(trades), 1)
        m["start"] = sig["date"].min()
        m["end"] = sig["date"].max()
        results[label] = m
        results[label]["pnl_df"] = pnl

    for label in ["IN-SAMPLE", "OUT-OF-SAMPLE"]:
        r = results[label]
        _log(f"\n  {label}  ({r.get('start','?')} → {r.get('end','?')})")
        _log(f"    Trades:     {r.get('n_trades',0)}")
        if r.get("n_trades", 0) > 0:
            _log(f"    Win Rate:   {r['win_rate']:.1%}")
            _log(f"    Total PnL:  ${r['total_pnl']:>12,.0f}")
            _log(f"    Ann Return: {r['ann_ret']:.2%}")
            _log(f"    Sharpe:     {r['sharpe']:.2f}")
            _log(f"    Sortino:    {r['sortino']:.2f}")
            _log(f"    Max DD:     {r['max_dd']:.2%}")

    return results, split_date


# ════════════════════════════════════════════════════════════
#  B.  WALK-FORWARD ANALYSIS
# ════════════════════════════════════════════════════════════

def walk_forward(signal_df, spx_df, n_windows=5):
    """Non-overlapping chronological windows."""
    _log("\n" + "=" * 65)
    _log(f"  B. WALK-FORWARD ANALYSIS ({n_windows} windows)")
    _log("=" * 65)

    dates = sorted(signal_df["date"].unique())
    chunk = len(dates) // n_windows
    results = []

    for i in range(n_windows):
        start = dates[i * chunk]
        end = dates[min((i + 1) * chunk - 1, len(dates) - 1)]
        window_sig = signal_df[(signal_df["date"] >= start) & (signal_df["date"] <= end)]
        trades, pnl = run_backtest(window_sig, spx_df)
        m = _metrics(_dr_for_metrics(pnl, spx_df)) if not pnl.empty else {}
        m["window"] = i + 1
        m["start"] = start
        m["end"] = end
        m["n_trades"] = len(trades)
        m["total_pnl"] = sum(t.net_pnl for t in trades) if trades else 0
        m["win_rate"] = (sum(1 for t in trades if t.net_pnl > 0)
                         / max(len(trades), 1)) if trades else 0
        m["pnl_df"] = pnl
        results.append(m)

        _log(f"\n  Window {i+1}: {start.date()} → {end.date()}")
        _log(f"    Trades={m['n_trades']}  Win={m['win_rate']:.0%}  "
             f"PnL=${m['total_pnl']:>10,.0f}  "
             f"Sharpe={m.get('sharpe', np.nan):>6.2f}  "
             f"MaxDD={m.get('max_dd', np.nan):.1%}")

    return results


# ════════════════════════════════════════════════════════════
#  C.  YEARLY BREAKDOWN
# ════════════════════════════════════════════════════════════

def yearly_breakdown(signal_df, spx_df):
    """Backtest each calendar year independently."""
    _log("\n" + "=" * 65)
    _log("  C. YEARLY PERFORMANCE BREAKDOWN")
    _log("=" * 65)

    signal_df = signal_df.copy()
    signal_df["year"] = signal_df["date"].dt.year
    years = sorted(signal_df["year"].unique())
    results = []

    for y in years:
        ysig = signal_df[signal_df["year"] == y]
        trades, pnl = run_backtest(ysig, spx_df)
        m = _metrics(_dr_for_metrics(pnl, spx_df)) if not pnl.empty else {}
        m["year"] = y
        m["n_trades"] = len(trades)
        m["total_pnl"] = sum(t.net_pnl for t in trades) if trades else 0
        m["win_rate"] = (sum(1 for t in trades if t.net_pnl > 0)
                         / max(len(trades), 1)) if trades else 0
        results.append(m)

    _log(f"\n  {'Year':>6s} {'Trades':>6s} {'Win%':>6s} {'TotalPnL':>14s} "
         f"{'Sharpe':>7s} {'MaxDD':>8s}")
    _log("  " + "─" * 55)
    for r in results:
        _log(f"  {r['year']:>6d} {r['n_trades']:>6d} {r['win_rate']:>5.0%} "
             f"${r['total_pnl']:>12,.0f} "
             f"{r.get('sharpe', np.nan):>7.2f} "
             f"{r.get('max_dd', np.nan):>7.1%}")

    return results


# ════════════════════════════════════════════════════════════
#  D.  STRESS-PERIOD ANALYSIS
# ════════════════════════════════════════════════════════════

STRESS_PERIODS = {
    "GFC (2008)":           ("2008-01-01", "2009-03-31"),
    "Flash Crash (2010)":   ("2010-04-01", "2010-07-31"),
    "EU Debt Crisis (2011)":("2011-07-01", "2011-12-31"),
    "Taper Tantrum (2013)": ("2013-05-01", "2013-09-30"),
    "China/Oil (2015–16)":  ("2015-08-01", "2016-02-29"),
    "Volmageddon (2018)":   ("2018-01-01", "2018-04-30"),
    "COVID Crash (2020)":   ("2020-02-01", "2020-05-31"),
    "Rate Hikes (2022)":    ("2022-01-01", "2022-12-31"),
}


def stress_test(signal_df, spx_df):
    """Performance during known market stress episodes."""
    _log("\n" + "=" * 65)
    _log("  D. STRESS-PERIOD ANALYSIS")
    _log("=" * 65)

    results = []
    for name, (s, e) in STRESS_PERIODS.items():
        window = signal_df[
            (signal_df["date"] >= s) & (signal_df["date"] <= e)
        ]
        trades, pnl = run_backtest(window, spx_df)
        m = _metrics(_dr_for_metrics(pnl, spx_df)) if not pnl.empty else {}
        m["period"] = name
        m["n_trades"] = len(trades)
        m["total_pnl"] = sum(t.net_pnl for t in trades) if trades else 0
        m["win_rate"] = (sum(1 for t in trades if t.net_pnl > 0)
                         / max(len(trades), 1)) if trades else 0
        results.append(m)

        _log(f"\n  {name}")
        _log(f"    Trades={m['n_trades']}  Win={m['win_rate']:.0%}  "
             f"PnL=${m['total_pnl']:>10,.0f}  "
             f"Sharpe={m.get('sharpe', np.nan):>6.2f}  "
             f"MaxDD={m.get('max_dd', np.nan):.1%}")

    return results


# ════════════════════════════════════════════════════════════
#  E.  VOLATILITY-REGIME ANALYSIS
# ════════════════════════════════════════════════════════════

def regime_analysis(signal_df, spx_df, features_df):
    """Split into high-vol and low-vol regimes based on trailing RV."""
    _log("\n" + "=" * 65)
    _log("  E. VOLATILITY-REGIME ANALYSIS")
    _log("=" * 65)

    rv_col = "rvol_monthly"
    if rv_col not in features_df.columns:
        rv_col = "rv_monthly"

    rv = features_df[["date", rv_col]].dropna()
    rv["rv_pctile"] = rv[rv_col].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    sig = signal_df.merge(rv[["date", "rv_pctile"]], on="date", how="left")

    regimes = {
        "Low Vol (bottom 33%)": sig[sig["rv_pctile"] <= 0.33],
        "Med Vol (middle 33%)": sig[(sig["rv_pctile"] > 0.33) & (sig["rv_pctile"] <= 0.66)],
        "High Vol (top 33%)":   sig[sig["rv_pctile"] > 0.66],
    }

    results = {}
    for label, sub in regimes.items():
        sub = sub.dropna(subset=["signal"])
        trades, pnl = run_backtest(sub, spx_df)
        m = _metrics(_dr_for_metrics(pnl, spx_df)) if not pnl.empty else {}
        m["n_trades"] = len(trades)
        m["total_pnl"] = sum(t.net_pnl for t in trades) if trades else 0
        m["win_rate"] = (sum(1 for t in trades if t.net_pnl > 0)
                         / max(len(trades), 1)) if trades else 0
        results[label] = m

        _log(f"\n  {label}")
        _log(f"    Trades={m['n_trades']}  Win={m['win_rate']:.0%}  "
             f"PnL=${m['total_pnl']:>10,.0f}  "
             f"Sharpe={m.get('sharpe', np.nan):>6.2f}  "
             f"MaxDD={m.get('max_dd', np.nan):.1%}")

    return results


# ════════════════════════════════════════════════════════════
#  F.  PARAMETER SENSITIVITY GRID
# ════════════════════════════════════════════════════════════

def param_sensitivity(features_df, spx_df):
    """Vary z-score threshold, holding period, and cost; report Sharpe."""
    _log("\n" + "=" * 65)
    _log("  F. PARAMETER SENSITIVITY GRID")
    _log("=" * 65)

    thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    hold_days = [1, 2, 3, 4, 5]  # weeklies: never hold past Friday expiry (max 5 trading days)
    costs = [0, 3, 5, 10, 15, 20]

    # F1: Threshold sensitivity (hold & cost at baseline)
    _log("\n  F1. Z-score Threshold Sensitivity")
    _log(f"  (hold={POSITION_HOLD_DAYS}d, cost={TRANSACTION_COST_BPS}bps)")
    thresh_results = []
    for thr in thresholds:
        sig = _build_vrp_signal_with_threshold(features_df, thr)
        trades, pnl = run_backtest(sig, spx_df)
        m = _metrics(_dr_for_metrics(pnl, spx_df)) if not pnl.empty else {}
        m["threshold"] = thr
        m["n_trades"] = len(trades)
        m["total_pnl"] = sum(t.net_pnl for t in trades) if trades else 0
        thresh_results.append(m)
        _log(f"    z>{thr:.2f}  Trades={m['n_trades']:>3d}  "
             f"Sharpe={m.get('sharpe', np.nan):>6.2f}  "
             f"PnL=${m['total_pnl']:>10,.0f}")

    # F2: Holding period sensitivity (threshold & cost at baseline)
    _log(f"\n  F2. Holding Period Sensitivity")
    _log(f"  (threshold={SIGNAL_ZSCORE_ENTRY}, cost={TRANSACTION_COST_BPS}bps)")
    hold_results = []
    sig_base = compute_vrp_signal(features_df)
    for hd in hold_days:
        trades, pnl = run_backtest(sig_base, spx_df, hold_days=hd)
        m = _metrics(_dr_for_metrics(pnl, spx_df)) if not pnl.empty else {}
        m["hold_days"] = hd
        m["n_trades"] = len(trades)
        m["total_pnl"] = sum(t.net_pnl for t in trades) if trades else 0
        hold_results.append(m)
        _log(f"    hold={hd:>2d}d  Trades={m['n_trades']:>3d}  "
             f"Sharpe={m.get('sharpe', np.nan):>6.2f}  "
             f"PnL=${m['total_pnl']:>10,.0f}")

    # F3: Transaction cost sensitivity
    _log(f"\n  F3. Transaction Cost Sensitivity")
    _log(f"  (threshold={SIGNAL_ZSCORE_ENTRY}, hold={POSITION_HOLD_DAYS}d)")
    cost_results = []
    for c in costs:
        trades, pnl = run_backtest(sig_base, spx_df, cost_bps=c)
        m = _metrics(_dr_for_metrics(pnl, spx_df)) if not pnl.empty else {}
        m["cost_bps"] = c
        m["n_trades"] = len(trades)
        m["total_pnl"] = sum(t.net_pnl for t in trades) if trades else 0
        cost_results.append(m)
        _log(f"    cost={c:>2d}bps  Trades={m['n_trades']:>3d}  "
             f"Sharpe={m.get('sharpe', np.nan):>6.2f}  "
             f"PnL=${m['total_pnl']:>10,.0f}")

    return thresh_results, hold_results, cost_results


def _build_vrp_signal_with_threshold(features_df, threshold):
    """Re-build VRP signal with a custom z-score threshold."""
    df = features_df.copy()
    rv_col = None
    for candidate in ["composite_rv_forecast", "har_rv_forecast", "garch_forecast"]:
        if candidate in df.columns and df[candidate].notna().sum() > 50:
            rv_col = candidate
            break
    if rv_col is None:
        return pd.DataFrame(columns=["date", "iv", "rv_forecast", "vrp", "vrp_zscore", "signal"])

    iv_col = "atm_iv_at_expiry"
    rv_vals = df[rv_col].dropna()
    iv_vals = df[iv_col].dropna()

    if rv_vals.median() > 2 * (iv_vals.median() ** 2):
        rv_numeric = pd.to_numeric(df[rv_col], errors="coerce").clip(lower=0)
        df["rv_forecast_vol"] = np.sqrt(rv_numeric.astype(float))
        rv_use = "rv_forecast_vol"
    else:
        rv_use = rv_col

    df["vrp"] = df[iv_col] - df[rv_use]
    df["vrp_mean"] = df["vrp"].rolling(SIGNAL_LOOKBACK, min_periods=60).mean()
    df["vrp_std"] = df["vrp"].rolling(SIGNAL_LOOKBACK, min_periods=60).std()
    df["vrp_zscore"] = (df["vrp"] - df["vrp_mean"]) / df["vrp_std"].replace(0, np.nan)

    df["signal"] = 0
    df.loc[df["vrp_zscore"] > threshold, "signal"] = 1
    df.loc[df["vrp_zscore"] < -threshold, "signal"] = -1

    out = df[["date", iv_col, rv_use, "vrp", "vrp_zscore", "signal"]].copy()
    out = out.rename(columns={iv_col: "iv", rv_use: "rv_forecast"})
    out = out.dropna(subset=["vrp_zscore"])
    return out


# ════════════════════════════════════════════════════════════
#  G.  BOOTSTRAP SHARPE CONFIDENCE INTERVALS
# ════════════════════════════════════════════════════════════

def bootstrap_sharpe(pnl_df, spx_df=None, n_boot=10000, ci=0.95):
    """Block-bootstrap the Sharpe ratio for statistical significance."""
    _log("\n" + "=" * 65)
    _log("  G. BOOTSTRAP SHARPE CONFIDENCE INTERVALS")
    _log("=" * 65)

    dr_series = _dr_for_metrics(pnl_df, spx_df)
    dr = dr_series.values if isinstance(dr_series, pd.Series) else dr_series
    n = len(dr)
    block_size = 22  # monthly blocks to preserve autocorrelation
    if n <= block_size:
        _log(f"  Skipping bootstrap: only {n} observations (need > {block_size}).")
        return np.array([]), np.nan, np.nan, np.nan

    n_blocks = max(n // block_size, 1)
    rng = np.random.RandomState(42)
    boot_sharpes = []

    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([dr[s:s + block_size] for s in starts])[:n]
        ann_r = ((1 + sample).prod()) ** (252 / len(sample)) - 1
        ann_v = sample.std() * np.sqrt(252)
        if ann_v > 0:
            boot_sharpes.append(ann_r / ann_v)

    boot_sharpes = np.array(boot_sharpes)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_sharpes, alpha * 100)
    hi = np.percentile(boot_sharpes, (1 - alpha) * 100)
    mean_s = boot_sharpes.mean()
    pct_positive = (boot_sharpes > 0).mean()

    _log(f"\n  Sample Sharpe:    {sharpe_ratio(dr_series):.2f}")
    _log(f"  Bootstrap Mean:   {mean_s:.2f}")
    _log(f"  95% CI:           [{lo:.2f}, {hi:.2f}]")
    _log(f"  P(Sharpe > 0):    {pct_positive:.1%}")

    return boot_sharpes, lo, hi, mean_s


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

def plot_oos_walkforward(oos_results, split_date, wf_results, spx_df=None):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # OOS cumulative PnL (strategy + benchmark)
    ax = fig.add_subplot(gs[0, 0])
    for label, color in [("IN-SAMPLE", PLOT_SECONDARY), ("OUT-OF-SAMPLE", PLOT_ACCENT)]:
        r = oos_results.get(label, {})
        pdf = r.get("pnl_df")
        if pdf is not None and not pdf.empty:
            ax.plot(pdf["date"], pdf["cumulative_pnl"] / 1e6,
                    color=color, lw=1.2, label=label)
    if spx_df is not None and not spx_df.empty:
        full_start = oos_results.get("IN-SAMPLE", {}).get("start")
        full_end = oos_results.get("OUT-OF-SAMPLE", {}).get("end")
        if full_start is not None and full_end is not None:
            bench = benchmark_returns_from_spx(spx_df, full_start, full_end)
            if len(bench) > 1:
                cum = (1 + bench).cumprod()
                ax.plot(bench.index, (cum - 1) * 1e6 / 1e6, color=PLOT_NEUTRAL, ls="--", lw=0.8, label="Buy & Hold SPX")
    ax.axvline(split_date, color=PLOT_NEUTRAL, ls="--", lw=1, label=f"Split: {split_date.date()}")
    ax.set_title("Out-of-Sample: Cumulative PnL ($M) vs SPX", fontweight="bold")
    ax.set_ylabel("$ Millions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # OOS metrics: Sharpe, Sortino, PSR
    ax = fig.add_subplot(gs[0, 1])
    labels_oos = ["IN-SAMPLE", "OUT-OF-SAMPLE"]
    metrics_to_show = ["sharpe", "sortino", "psr"]
    x = np.arange(len(metrics_to_show))
    w = 0.35
    for i, label in enumerate(labels_oos):
        r = oos_results.get(label, {})
        vals = [r.get(m, 0) if not (isinstance(r.get(m), float) and np.isnan(r.get(m))) else 0 for m in metrics_to_show]
        ax.bar(x + i * w, vals, w, label=label,
               color=[PLOT_SECONDARY, PLOT_ACCENT][i], alpha=0.8)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(["Sharpe", "Sortino", "PSR"])
    ax.set_title("OOS Risk-Adjusted Metrics (PSR = Prob. Sharpe > 0)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Walk-forward Sharpe by window
    ax = fig.add_subplot(gs[1, 0])
    wf_sharpes = [r.get("sharpe", 0) for r in wf_results]
    wf_labels = [f"W{r['window']}\n{r['start'].strftime('%Y-%m')}" for r in wf_results]
    colors = [PLOT_POSITIVE if s > 0 else PLOT_NEGATIVE for s in wf_sharpes]
    ax.bar(wf_labels, wf_sharpes, color=colors, alpha=0.7, edgecolor="none")
    ax.set_title("Walk-Forward: Sharpe by Window", fontweight="bold")
    ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Walk-forward cumulative PnL overlaid
    ax = fig.add_subplot(gs[1, 1])
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(wf_results)))
    for r, c in zip(wf_results, cmap):
        pdf = r.get("pnl_df")
        if pdf is not None and not pdf.empty:
            ax.plot(pdf["date"], pdf["cumulative_pnl"] / 1e6,
                    color=c, lw=1, label=f"W{r['window']}")
    ax.set_title("Walk-Forward: Cumulative PnL ($M)", fontweight="bold")
    ax.set_ylabel("$ Millions")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Out-of-Sample & Walk-Forward Analysis (vs Buy & Hold SPX)",
                 fontsize=15, fontweight="bold", y=1.01)
    _save(fig, "15_oos_walkforward")


def plot_stress_periods(stress_results, spx_df=None):
    n_cols = 4 if spx_df is not None and not spx_df.empty else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    names = [r["period"] for r in stress_results]
    x = np.arange(len(names))

    # Sharpe
    ax = axes[0]
    sharpes = [r.get("sharpe", 0) for r in stress_results]
    colors = [PLOT_POSITIVE if s > 0 else PLOT_NEGATIVE for s in sharpes]
    ax.barh(x, sharpes, color=colors, alpha=0.7, edgecolor="none")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Sharpe Ratio", fontweight="bold")
    ax.axvline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Total PnL
    ax = axes[1]
    pnls = [r.get("total_pnl", 0) / 1e6 for r in stress_results]
    colors = [PLOT_POSITIVE if p > 0 else PLOT_NEGATIVE for p in pnls]
    ax.barh(x, pnls, color=colors, alpha=0.7, edgecolor="none")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Total PnL ($M)", fontweight="bold")
    ax.axvline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Max Drawdown
    ax = axes[2]
    dds = [r.get("max_dd", 0) * 100 for r in stress_results]
    ax.barh(x, dds, color=PLOT_COLORS["600"], alpha=0.6, edgecolor="none")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Max Drawdown (%)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Benchmark (SPX) return during each stress period
    if n_cols == 4 and spx_df is not None:
        ax = axes[3]
        bench_rets = []
        for r in stress_results:
            period = r["period"]
            start, end = STRESS_PERIODS.get(period, (None, None))
            if start and end:
                bench = benchmark_returns_from_spx(spx_df, start, end)
                tot_ret = (1 + bench).prod() - 1 if len(bench) > 0 else 0
                bench_rets.append(tot_ret * 100)
            else:
                bench_rets.append(0)
        colors_b = [PLOT_POSITIVE if b > 0 else PLOT_NEGATIVE for b in bench_rets]
        ax.barh(x, bench_rets, color=colors_b, alpha=0.6, edgecolor="none")
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title("SPX Return (%)", fontweight="bold")
        ax.axvline(0, color=PLOT_NEUTRAL, lw=0.5)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Stress-Period Performance (Strategy vs SPX)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "16_stress_periods")


def plot_regime_analysis(regime_results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    names = list(regime_results.keys())
    x = np.arange(len(names))

    for ax, metric, title in zip(
        axes,
        ["sharpe", "total_pnl", "win_rate", "psr"],
        ["Sharpe Ratio", "Total PnL ($K)", "Win Rate (%)", "PSR (Prob. Sharpe > 0)"],
    ):
        vals = []
        for n in names:
            v = regime_results[n].get(metric, 0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0
            if metric == "total_pnl":
                v /= 1000
            elif metric == "win_rate":
                v *= 100
            elif metric == "psr":
                v = v * 100 if not np.isnan(v) else 0
            vals.append(v)
        colors = PLOT_PALETTE[:4]
        ax.bar(x, vals, color=colors[:len(names)], alpha=0.8, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Volatility-Regime Analysis (incl. PSR)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "17_regime_analysis")


def plot_param_sensitivity(thresh_res, hold_res, cost_res):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Threshold (Sharpe + PSR on secondary)
    ax = axes[0]
    thrs = [r["threshold"] for r in thresh_res]
    shs = [r.get("sharpe", 0) for r in thresh_res]
    psrs = [r.get("psr", np.nan) * 100 if not (isinstance(r.get("psr"), float) and np.isnan(r.get("psr"))) else 0 for r in thresh_res]
    nt = [r.get("n_trades", 0) for r in thresh_res]
    x = np.arange(len(thrs))
    ax.bar(x - 0.2, shs, 0.35, color=PLOT_SECONDARY, alpha=0.7, label="Sharpe")
    ax.set_xticks(x)
    ax.set_xticklabels([f"z>{t:.1f}" for t in thrs], fontsize=9)
    ax.set_title("Z-Score Threshold (Sharpe & PSR)", fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, psrs, "go-", ms=6, label="PSR (%)", color=PLOT_POSITIVE)
    ax2.set_ylabel("PSR (%)", color=PLOT_POSITIVE)
    ax2.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # Holding period (max 5d for weeklies; backtest exits at expiry)
    ax = axes[1]
    hds = [r["hold_days"] for r in hold_res]
    shs = [r.get("sharpe", 0) for r in hold_res]
    pnls = [r.get("total_pnl", 0) / 1e6 for r in hold_res]
    ax.bar(np.arange(len(hds)), shs, color=PLOT_ACCENT, alpha=0.7)
    ax.set_xticks(np.arange(len(hds)))
    ax.set_xticklabels([f"{h}d" for h in hds], fontsize=9)
    ax.set_title("Holding Period (≤ expiry, max 5d)", fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(hds)), pnls, "go-", ms=6, label="PnL ($M)")
    ax2.set_ylabel("Total PnL ($M)", color=PLOT_POSITIVE)
    ax.grid(True, alpha=0.3, axis="y")

    # Transaction cost
    ax = axes[2]
    cs = [r["cost_bps"] for r in cost_res]
    shs = [r.get("sharpe", 0) for r in cost_res]
    pnls = [r.get("total_pnl", 0) / 1e6 for r in cost_res]
    ax.bar(np.arange(len(cs)), shs, color=PLOT_COLORS["300"], alpha=0.7)
    ax.set_xticks(np.arange(len(cs)))
    ax.set_xticklabels([f"{c}bp" for c in cs], fontsize=9)
    ax.set_title("Transaction Cost", fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(cs)), pnls, "go-", ms=6, label="PnL ($M)")
    ax2.set_ylabel("Total PnL ($M)", color=PLOT_POSITIVE)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Parameter Sensitivity (incl. PSR = Prob. Sharpe > 0)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "18_param_sensitivity")


def plot_bootstrap(boot_sharpes, lo, hi, mean_s, psr=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(boot_sharpes, bins=80, color=PLOT_SECONDARY, alpha=0.6, edgecolor="none",
            density=True, label=f"Bootstrap (n={len(boot_sharpes):,})")
    ax.axvline(mean_s, color=PLOT_COLORS["800"], lw=2, ls="--", label=f"Mean = {mean_s:.2f}")
    ax.axvline(lo, color=PLOT_PRIMARY, lw=1.5, ls=":", label=f"95% CI lower = {lo:.2f}")
    ax.axvline(hi, color=PLOT_PRIMARY, lw=1.5, ls=":", label=f"95% CI upper = {hi:.2f}")
    ax.axvline(0, color=PLOT_NEUTRAL, lw=1, ls="-", alpha=0.5)

    pct_pos = (boot_sharpes > 0).mean()
    txt = f"P(Sharpe > 0) = {pct_pos:.1%}"
    if psr is not None and not (isinstance(psr, float) and np.isnan(psr)):
        txt += f"\nPSR (Prob. Sharpe > 0) = {psr:.1%}"
    ax.text(0.02, 0.95, txt,
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    ax.set_title("Bootstrap Distribution of Sharpe Ratio (Block Bootstrap, 10k draws)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save(fig, "19_bootstrap_sharpe")


def plot_yearly(yearly_results, spx_df=None):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    years = [r["year"] for r in yearly_results]
    x = np.arange(len(years))
    w = 0.35

    # Sharpe by year (optionally vs benchmark)
    ax = axes[0, 0]
    shs = [r.get("sharpe", 0) for r in yearly_results]
    colors = [PLOT_POSITIVE if s > 0 else PLOT_NEGATIVE for s in shs]
    ax.bar(x - w / 2, shs, w, color=colors, alpha=0.7, edgecolor="none", label="Strategy")
    if spx_df is not None and not spx_df.empty:
        bench_sharpes = []
        for r in yearly_results:
            y = r["year"]
            start, end = f"{y}-01-01", f"{y}-12-31"
            bench = benchmark_returns_from_spx(spx_df, start, end)
            if len(bench) > 10:
                sr_b = sharpe_ratio(bench)
                bench_sharpes.append(sr_b)
            else:
                bench_sharpes.append(0)
        ax.bar(x + w / 2, bench_sharpes, w, color=PLOT_NEUTRAL, alpha=0.6, label="SPX")
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=8, rotation=45)
    ax.set_title("Sharpe Ratio by Year (vs SPX)", fontweight="bold")
    ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Total PnL by year
    ax = axes[0, 1]
    pnls = [r.get("total_pnl", 0) / 1e6 for r in yearly_results]
    colors = [PLOT_POSITIVE if p > 0 else PLOT_NEGATIVE for p in pnls]
    ax.bar(x, pnls, color=colors, alpha=0.7, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=8, rotation=45)
    ax.set_title("Total PnL ($M) by Year", fontweight="bold")
    ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Win rate by year
    ax = axes[1, 0]
    wrs = [r.get("win_rate", 0) * 100 for r in yearly_results]
    ax.bar(x, wrs, color=PLOT_SECONDARY, alpha=0.7, edgecolor="none")
    ax.axhline(50, color=PLOT_NEUTRAL, ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=8, rotation=45)
    ax.set_title("Win Rate (%) by Year", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    # Max drawdown by year
    ax = axes[1, 1]
    dds = [r.get("max_dd", 0) * 100 for r in yearly_results]
    ax.bar(x, dds, color=PLOT_COLORS["600"], alpha=0.6, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=8, rotation=45)
    ax.set_title("Max Drawdown (%) by Year", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Yearly Performance Breakdown (Strategy vs SPX)",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "20_yearly_breakdown")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    _log("\n" + "=" * 65)
    _log("  ROBUSTNESS, OUT-OF-SAMPLE & STRESS TESTING SUITE")
    _log("=" * 65)
    _log("  (All backtests hold only until option expiry; no holding past expiry.)")

    # ── Load & prepare data ────────────────────────────────
    _log("\nLoading data ...")
    data = load_all_data()
    _log("Building features ...")
    features = build_feature_table(
        data["options"], data["spx"], rf_series=data["rf"]
    )
    _log("Running RV models ...")
    forecasts = run_all_rv_models(features, train_window=252)
    augmented = features.merge(forecasts, on="date", how="left")

    _log("Building VRP signal ...")
    signal_df = compute_vrp_signal(augmented)
    spx_df = data["spx"]

    # Full-sample backtest for baseline
    _log("\nFull-sample baseline backtest ...")
    full_trades, full_pnl = run_backtest(signal_df, spx_df)
    full_m = _metrics(_dr_for_metrics(full_pnl, spx_df))
    _log(f"  Baseline: {len(full_trades)} trades, "
         f"Sharpe={full_m['sharpe']:.2f}, MaxDD={full_m['max_dd']:.1%}")

    # ── A. Out-of-sample ──────────────────────────────────
    oos_res, split_date = oos_test(signal_df, spx_df)

    # ── B. Walk-forward ───────────────────────────────────
    wf_res = walk_forward(signal_df, spx_df, n_windows=5)

    # ── C. Yearly breakdown ───────────────────────────────
    yearly_res = yearly_breakdown(signal_df, spx_df)

    # ── D. Stress periods ─────────────────────────────────
    stress_res = stress_test(signal_df, spx_df)

    # ── E. Vol-regime analysis ────────────────────────────
    regime_res = regime_analysis(signal_df, spx_df, augmented)

    # ── F. Parameter sensitivity ──────────────────────────
    thresh_res, hold_res, cost_res = param_sensitivity(augmented, spx_df)

    # ── G. Bootstrap Sharpe ───────────────────────────────
    boot_sharpes, lo, hi, mean_s = bootstrap_sharpe(full_pnl, spx_df=spx_df)

    # ── Save report ───────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "robustness_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(REPORT))
    _log(f"\n  Report saved → {report_path}")

    # ── Generate plots ────────────────────────────────────
    _log("\n  Generating plots ...")
    plot_oos_walkforward(oos_res, split_date, wf_res, spx_df=spx_df)
    plot_stress_periods(stress_res, spx_df=spx_df)
    plot_regime_analysis(regime_res)
    plot_param_sensitivity(thresh_res, hold_res, cost_res)
    full_psr = probabilistic_sharpe_ratio(_dr_for_metrics(full_pnl, spx_df), sr_ref=0.0) if not full_pnl.empty else None
    if len(boot_sharpes) > 0:
        plot_bootstrap(boot_sharpes, lo, hi, mean_s, psr=full_psr)
    plot_yearly(yearly_res, spx_df=spx_df)

    _log("\n" + "=" * 65)
    _log("  ROBUSTNESS SUITE COMPLETE")
    _log("=" * 65 + "\n")


if __name__ == "__main__":
    main()
