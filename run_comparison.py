#!/usr/bin/env python3
# ============================================================
# run_comparison.py — Head-to-Head Strategy Comparison
# ============================================================
"""
Compare the three signal families from the proposal:

    Strategy 1 — VRP (Core):  ATM IV − forecast RV
    Strategy 2 — Variance Swap:  Model-free IV (MFIV) − forecast RV
    Strategy 3 — Distribution:  RN vs physical tail divergence

Also analyses FOMC-window vs non-FOMC performance for each.

Produces:
    output/figures/13_strategy_comparison.png
    output/figures/14_fomc_analysis.png
    output/reports/strategy_comparison.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR,
    SIGNAL_ZSCORE_ENTRY, SIGNAL_LOOKBACK,
    NOTIONAL_CAPITAL, TRADING_DAYS_PER_YEAR,
)
from src.data_loader import load_all_data
from src.feature_engineering import build_feature_table
from src.rv_models import run_all_rv_models
from src.signals import (
    compute_vrp_signal,
    compute_skew_signal,
    compute_term_structure_signal,
    compute_distribution_signal,
)
from src.backtest import run_backtest, trades_to_dataframe
from src.performance import (
    sharpe_ratio, sortino_ratio, annualised_return,
    annualised_volatility, compute_drawdown, return_statistics,
)

warnings.filterwarnings("ignore")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-v0_8")


# ════════════════════════════════════════════════════════════
#  SIGNAL BUILDERS — one per strategy
# ════════════════════════════════════════════════════════════

def _make_vrp_signal(feature_df):
    """Strategy 1: ATM IV − forecast RV."""
    return compute_vrp_signal(feature_df, iv_col="atm_iv_30d")


def _make_varswap_signal(feature_df):
    """
    Strategy 2: Model-free IV (variance swap) − forecast RV.
    Uses mfiv_vol_30d as the IV measure instead of ATM IV.
    """
    if "mfiv_vol_30d" not in feature_df.columns:
        print("[comparison] mfiv_vol_30d not available — skipping var-swap strategy.")
        return None

    df = feature_df.dropna(subset=["mfiv_vol_30d"])
    if len(df) < 100:
        print("[comparison] Not enough MFIV data — skipping var-swap strategy.")
        return None

    return compute_vrp_signal(df, iv_col="mfiv_vol_30d")


def _make_distribution_signal(feature_df, spx_df, options_df):
    """
    Strategy 3: Distribution-based.
    Convert the continuous dist_signal into a discrete ±1/0 trading signal
    using z-score thresholding (same framework as VRP).
    """
    dist_raw = compute_distribution_signal(spx_df, options_df)
    if dist_raw.empty or len(dist_raw) < 100:
        print("[comparison] Not enough distribution signal data.")
        return None

    df = dist_raw.copy()
    df["dist_mean"] = df["dist_signal"].rolling(SIGNAL_LOOKBACK, min_periods=60).mean()
    df["dist_std"] = df["dist_signal"].rolling(SIGNAL_LOOKBACK, min_periods=60).std()
    df["vrp_zscore"] = (
        (df["dist_signal"] - df["dist_mean"])
        / df["dist_std"].replace(0, np.nan)
    )

    # Signal: positive dist_signal (RN tails heavy) → short vol (sell overpriced protection)
    #         negative dist_signal (RN tails light) → long vol  (buy cheap protection)
    df["signal"] = 0
    df.loc[df["vrp_zscore"] > SIGNAL_ZSCORE_ENTRY, "signal"] = 1    # short vol
    df.loc[df["vrp_zscore"] < -SIGNAL_ZSCORE_ENTRY, "signal"] = -1  # long vol

    # Need iv and rv_forecast columns for the backtester
    # Merge from the feature table
    feat = feature_df[["date", "atm_iv_30d"]].dropna()
    df = df.merge(feat, on="date", how="left")
    df = df.rename(columns={"atm_iv_30d": "iv"})
    df["rv_forecast"] = df["iv"]  # placeholder
    df["vrp"] = df["dist_signal"]

    df = df.dropna(subset=["vrp_zscore", "iv"])

    n_short = (df["signal"] == 1).sum()
    n_long = (df["signal"] == -1).sum()
    print(f"[comparison] Distribution signal: {len(df)} days — "
          f"short_vol={n_short}, long_vol={n_long}")

    return df[["date", "iv", "rv_forecast", "vrp", "vrp_zscore", "signal"]]


# ════════════════════════════════════════════════════════════
#  PERFORMANCE SUMMARY
# ════════════════════════════════════════════════════════════

def _summarise(name, trades, pnl_df):
    """Compute a summary dict for one strategy."""
    if pnl_df.empty or len(pnl_df) < 5:
        return {"strategy": name, "n_trades": 0}

    dr = pnl_df["daily_return"]
    trades_df = trades_to_dataframe(trades)

    wins = sum(1 for t in trades if t.net_pnl > 0)
    total_pnl = sum(t.net_pnl for t in trades)
    dd = compute_drawdown(dr)

    return {
        "strategy": name,
        "n_trades": len(trades),
        "win_rate": wins / max(len(trades), 1),
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / max(len(trades), 1),
        "ann_return": annualised_return(dr),
        "ann_vol": annualised_volatility(dr),
        "sharpe": sharpe_ratio(dr),
        "sortino": sortino_ratio(dr),
        "max_dd": dd["max_drawdown"],
        "skewness": dr.skew(),
        "kurtosis": dr.kurtosis(),
    }


# ════════════════════════════════════════════════════════════
#  FOMC ANALYSIS
# ════════════════════════════════════════════════════════════

def _fomc_split(signal_df, spx_df, feature_df, strategy_name):
    """
    Split a strategy's trades into FOMC-window and non-FOMC-window
    and compare performance.
    """
    # Merge fomc_window flag into signal_df
    fomc_flags = feature_df[["date", "fomc_window"]].drop_duplicates("date")
    merged = signal_df.merge(fomc_flags, on="date", how="left")
    merged["fomc_window"] = merged["fomc_window"].fillna(0).astype(int)

    results = {}
    for label, flag in [("fomc", 1), ("non_fomc", 0)]:
        sub = merged[merged["fomc_window"] == flag].copy()
        if sub["signal"].abs().sum() == 0:
            results[label] = {"n_trades": 0}
            continue
        trades, pnl_df = run_backtest(sub, spx_df)
        results[label] = _summarise(f"{strategy_name}_{label}", trades, pnl_df)

    return results


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [viz] Saved → {path.relative_to(FIGURES_DIR.parent.parent)}")
    plt.close(fig)


def plot_strategy_comparison(results, pnl_dict):
    """5-panel comparison of the three strategies."""
    strategies = [r["strategy"] for r in results if r["n_trades"] > 0]
    if len(strategies) == 0:
        print("  [viz] No strategies to compare — skipping.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # 1. Cumulative PnL
    ax = axes[0, 0]
    for name in strategies:
        if name in pnl_dict and not pnl_dict[name].empty:
            pdf = pnl_dict[name]
            ax.plot(pdf["date"], pdf["cumulative_pnl"] / 1e6, label=name, lw=1.2)
    ax.set_title("Cumulative PnL ($M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("$ Millions")
    ax.legend(fontsize=9)
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(True, alpha=0.3)

    # 2. Sharpe ratios
    ax = axes[0, 1]
    sharpes = [r["sharpe"] for r in results if r["n_trades"] > 0]
    colors = ["green" if s > 0 else "red" for s in sharpes]
    ax.bar(strategies, sharpes, color=colors, alpha=0.7, edgecolor="none")
    ax.set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Win Rate & # Trades
    ax = axes[0, 2]
    n_trades = [r["n_trades"] for r in results if r["n_trades"] > 0]
    win_rates = [r["win_rate"] * 100 for r in results if r["n_trades"] > 0]
    x = np.arange(len(strategies))
    bars = ax.bar(x, n_trades, color="steelblue", alpha=0.7, label="# Trades")
    ax2 = ax.twinx()
    ax2.plot(x, win_rates, "ro-", label="Win Rate %", ms=8)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_title("Trades & Win Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Trades")
    ax2.set_ylabel("Win Rate %")
    ax2.set_ylim(30, 80)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Drawdown comparison
    ax = axes[1, 0]
    for name in strategies:
        if name in pnl_dict and not pnl_dict[name].empty:
            pdf = pnl_dict[name]
            cum = (1 + pdf["daily_return"]).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            ax.plot(pdf["date"], dd * 100, label=name, lw=0.8)
    ax.set_title("Drawdown (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown %")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. Return distribution
    ax = axes[1, 1]
    for name in strategies:
        if name in pnl_dict and not pnl_dict[name].empty:
            pdf = pnl_dict[name]
            ax.hist(pdf["daily_return"] * 100, bins=50, alpha=0.4, label=name)
    ax.set_title("Daily Return Distribution (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Daily Return %")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for r in results:
        if r["n_trades"] > 0:
            table_data.append([
                r["strategy"],
                f"{r['n_trades']}",
                f"{r['win_rate']:.0%}",
                f"${r['total_pnl']/1e6:.2f}M",
                f"{r['sharpe']:.2f}",
                f"{r['sortino']:.2f}",
                f"{r['max_dd']:.1%}",
            ])
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["Strategy", "Trades", "Win%", "Total PnL", "Sharpe", "Sortino", "Max DD"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Performance Summary", fontsize=12, fontweight="bold")

    fig.suptitle("Strategy Comparison: VRP vs Variance Swap vs Distribution",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "13_strategy_comparison")


def plot_fomc_analysis(fomc_results):
    """Compare FOMC vs non-FOMC performance across strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    strategies = list(fomc_results.keys())
    metrics = ["sharpe", "total_pnl", "win_rate"]
    titles = ["Sharpe Ratio", "Total PnL ($K)", "Win Rate (%)"]

    for ax, metric, title in zip(axes, metrics, titles):
        fomc_vals = []
        nonfomc_vals = []
        labels = []

        for strat in strategies:
            fr = fomc_results[strat]
            f_val = fr.get("fomc", {}).get(metric, 0)
            nf_val = fr.get("non_fomc", {}).get(metric, 0)

            if metric == "total_pnl":
                f_val /= 1000
                nf_val /= 1000
            elif metric == "win_rate":
                f_val *= 100
                nf_val *= 100

            fomc_vals.append(f_val)
            nonfomc_vals.append(nf_val)
            labels.append(strat)

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, fomc_vals, w, label="FOMC Window (±7d)",
               color="coral", alpha=0.8, edgecolor="none")
        ax.bar(x + w / 2, nonfomc_vals, w, label="Non-FOMC",
               color="steelblue", alpha=0.8, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.axhline(0, color="grey", lw=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("FOMC Window (±7 days) vs Non-FOMC Performance",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "14_fomc_analysis")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  STRATEGY COMPARISON — VRP vs Variance Swap vs Distribution")
    print("  + FOMC Window Analysis (±7 days)")
    print("=" * 65 + "\n")

    # ── Load data ──────────────────────────────────────────
    print("Loading data ...")
    data = load_all_data()
    print("Building features ...")
    features = build_feature_table(
        data["options"], data["spx"], rf_series=data["rf"]
    )
    print("Running RV models ...")
    forecasts = run_all_rv_models(features, train_window=252)
    augmented = features.merge(forecasts, on="date", how="left")

    # ── Build signals for each strategy ────────────────────
    print("\n" + "─" * 50)
    print("  Building signals for each strategy ...")
    print("─" * 50)

    sig_vrp = _make_vrp_signal(augmented)
    sig_varswap = _make_varswap_signal(augmented)
    sig_dist = _make_distribution_signal(augmented, data["spx"], data["options"])

    # ── Backtest each strategy ─────────────────────────────
    print("\n" + "─" * 50)
    print("  Backtesting each strategy ...")
    print("─" * 50)

    results = []
    pnl_dict = {}

    for name, sig_df in [
        ("VRP (ATM IV)", sig_vrp),
        ("Var Swap (MFIV)", sig_varswap),
        ("Distribution", sig_dist),
    ]:
        if sig_df is None or sig_df.empty:
            results.append({"strategy": name, "n_trades": 0})
            continue

        print(f"\n  → {name}")
        trades, pnl_df = run_backtest(sig_df, data["spx"])
        results.append(_summarise(name, trades, pnl_df))
        pnl_dict[name] = pnl_df

    # ── FOMC analysis ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("  FOMC Window Analysis (±7 days) ...")
    print("─" * 50)

    fomc_results = {}
    for name, sig_df in [
        ("VRP (ATM IV)", sig_vrp),
        ("Var Swap (MFIV)", sig_varswap),
        ("Distribution", sig_dist),
    ]:
        if sig_df is None or sig_df.empty:
            continue
        print(f"\n  → {name}")
        fomc_results[name] = _fomc_split(
            sig_df, data["spx"], augmented, name
        )

    # ── Print report ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STRATEGY COMPARISON RESULTS")
    print("=" * 65)

    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("  Strategy Comparison — VRP vs Variance Swap vs Distribution")
    report_lines.append("=" * 65)
    report_lines.append("")

    for r in results:
        name = r["strategy"]
        report_lines.append(f"  {name}")
        report_lines.append("  " + "─" * 40)
        if r["n_trades"] == 0:
            report_lines.append("    No trades generated.")
            print(f"\n  {name}: No trades generated.")
        else:
            lines = [
                f"    Trades:        {r['n_trades']}",
                f"    Win Rate:      {r['win_rate']:.1%}",
                f"    Total PnL:     ${r['total_pnl']:,.0f}",
                f"    Avg PnL:       ${r['avg_pnl']:,.0f}",
                f"    Ann. Return:   {r['ann_return']:.2%}",
                f"    Ann. Vol:      {r['ann_vol']:.2%}",
                f"    Sharpe:        {r['sharpe']:.2f}",
                f"    Sortino:       {r['sortino']:.2f}",
                f"    Max Drawdown:  {r['max_dd']:.2%}",
                f"    Skewness:      {r['skewness']:.3f}",
                f"    Kurtosis:      {r['kurtosis']:.3f}",
            ]
            for l in lines:
                report_lines.append(l)
                print(l)
        report_lines.append("")

    # FOMC section
    report_lines.append("")
    report_lines.append("=" * 65)
    report_lines.append("  FOMC Window (±7 days) vs Non-FOMC")
    report_lines.append("=" * 65)
    print("\n" + "=" * 65)
    print("  FOMC Window (±7 days) vs Non-FOMC")
    print("=" * 65)

    for strat, splits in fomc_results.items():
        report_lines.append(f"\n  {strat}")
        print(f"\n  {strat}")
        for label in ["fomc", "non_fomc"]:
            s = splits.get(label, {})
            tag = "FOMC (±7d)" if label == "fomc" else "Non-FOMC"
            if s.get("n_trades", 0) == 0:
                msg = f"    {tag:15s}  No trades"
                report_lines.append(msg)
                print(msg)
            else:
                msg = (
                    f"    {tag:15s}  Trades={s['n_trades']:>3d}  "
                    f"Win={s['win_rate']:.0%}  "
                    f"PnL=${s['total_pnl']:>12,.0f}  "
                    f"Sharpe={s['sharpe']:>6.2f}  "
                    f"MaxDD={s['max_dd']:.1%}"
                )
                report_lines.append(msg)
                print(msg)

    # ── Save report ────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "strategy_comparison.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved → {report_path}")

    # ── Plots ──────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_strategy_comparison(results, pnl_dict)
    if fomc_results:
        plot_fomc_analysis(fomc_results)

    print("\n" + "=" * 65)
    print("  COMPARISON COMPLETE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
