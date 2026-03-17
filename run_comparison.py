#!/usr/bin/env python3
"""
Strategy comparison: VRP (ATM IV − RV) vs Variance Swap (MFIV − RV).
FOMC-window analysis. Outputs figures 13–14 and strategy_comparison.txt.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    REPORTS_DIR,
    SIGNAL_ZSCORE_ENTRY, SIGNAL_LOOKBACK,
    NOTIONAL_CAPITAL, TRADING_DAYS_PER_YEAR,
    PLOT_COLORS, PLOT_PALETTE, PLOT_PRIMARY, PLOT_SECONDARY,
    PLOT_ACCENT, PLOT_NEUTRAL, PLOT_POSITIVE, PLOT_NEGATIVE,
)
from src.pipeline import load_data_and_augment
from src.signals import compute_vrp_signal
from src.backtest import run_backtest, trades_to_dataframe
from src.visualization import _save
from src.performance import (
    compute_drawdown, return_statistics, probabilistic_sharpe_ratio,
    benchmark_returns_from_spx, trade_statistics,
    realized_returns_from_trades,
    _realized_ann_return, _realized_ann_vol, _realized_sharpe, _realized_sortino,
    _benchmark_comparison_realized,
)

warnings.filterwarnings("ignore")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-v0_8")


def _fmt_alpha(a):
    """Format alpha for display; avoid absurd values from mis-scaled regressions."""
    if np.isnan(a) or np.isinf(np.abs(a)) or np.abs(a) > 10:
        return "N/A"
    return f"{a:.2%}"


def _make_vrp_signal(feature_df):
    """Strategy 1: ATM IV − forecast RV."""
    return compute_vrp_signal(feature_df, iv_col="atm_iv_at_expiry")


def _make_varswap_signal(feature_df):
    """
    Strategy 2: Model-free IV (variance swap) − forecast RV.
    Uses mfiv_vol_at_expiry as the IV measure instead of ATM IV.
    """
    if "mfiv_vol_at_expiry" not in feature_df.columns:
        print("[comparison] mfiv_vol_at_expiry not available — skipping var-swap strategy.")
        return None

    df = feature_df.dropna(subset=["mfiv_vol_at_expiry"])
    if len(df) < 100:
        print("[comparison] Not enough MFIV data — skipping var-swap strategy.")
        return None

    return compute_vrp_signal(df, iv_col="mfiv_vol_at_expiry")


def _summarise(name, trades, pnl_df, spx_df=None):
    """Compute a summary dict for one strategy (realized returns only)."""
    trades_df = trades_to_dataframe(trades)
    if pnl_df.empty or len(trades) < 1:
        return {"strategy": name, "n_trades": 0}

    dr = realized_returns_from_trades(trades_df, spx_df)
    if dr.empty or dr.dropna().shape[0] < 2:
        return {"strategy": name, "n_trades": len(trades)}

    wins = sum(1 for t in trades if t.net_pnl > 0)
    total_pnl = sum(t.net_pnl for t in trades)
    dd = compute_drawdown(dr, sparse_realized=True)
    dr_exits = dr.dropna()

    out = {
        "strategy": name,
        "n_trades": len(trades),
        "win_rate": wins / max(len(trades), 1),
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / max(len(trades), 1),
        "ann_return": _realized_ann_return(dr),
        "ann_vol": _realized_ann_vol(dr),
        "sharpe": _realized_sharpe(dr),
        "sortino": _realized_sortino(dr),
        "max_dd": dd["max_drawdown"],
        "skewness": dr_exits.skew(),
        "kurtosis": dr_exits.kurtosis(),
        "psr": probabilistic_sharpe_ratio(dr_exits, sr_ref=0.0),
    }
    if not trades_df.empty:
        ts = trade_statistics(trades_df)
        out["avg_holding_days"] = ts.get("avg_holding_days", np.nan)
        out["trades_per_year"] = ts.get("trades_per_year", np.nan)
    else:
        out["avg_holding_days"] = np.nan
        out["trades_per_year"] = np.nan

    if spx_df is not None and len(dr_exits) > 10:
        start_date, end_date = dr.index.min(), dr.index.max()
        bench_ret = benchmark_returns_from_spx(spx_df, start_date, end_date)
        if len(bench_ret) > 0:
            bc = _benchmark_comparison_realized(dr, bench_ret)
            out["benchmark_sharpe"] = bc.get("benchmark_sharpe", np.nan)
            out["benchmark_ann_return"] = bc.get("benchmark_ann_return", np.nan)
            out["alpha_ann"] = bc.get("alpha_ann", np.nan)
            out["information_ratio"] = bc.get("information_ratio", np.nan)
    return out


def _fomc_split(signal_df, spx_df, feature_df, strategy_name):
    """
    Split a strategy's trades into FOMC-window and non-FOMC-window
    and compare performance (incl. PSR when spx_df available).
    """
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
        results[label] = _summarise(f"{strategy_name}_{label}", trades, pnl_df, spx_df=spx_df)

    return results


def plot_strategy_comparison(results, pnl_dict, spx_df=None):
    """6-panel comparison: strategies + benchmark (SPX buy-and-hold), PSR and Alpha in table."""
    strategies = [r["strategy"] for r in results if r["n_trades"] > 0]
    if len(strategies) == 0:
        print("  [viz] No strategies to compare — skipping.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # 1. Cumulative PnL (strategies + benchmark)
    ax = axes[0, 0]
    for name in strategies:
        if name in pnl_dict and not pnl_dict[name].empty:
            pdf = pnl_dict[name]
            cum_pnl = pdf["daily_pnl"].fillna(0).cumsum()
            ax.plot(pdf["date"], cum_pnl / 1e6, label=name, lw=1.2)
    # Add buy-and-hold SPX benchmark ($1M notional)
    if spx_df is not None and not spx_df.empty:
        all_dates = pd.concat([pnl_dict[n]["date"] for n in strategies if n in pnl_dict and not pnl_dict[n].empty], ignore_index=True)
        start, end = all_dates.min(), all_dates.max()
        spx = spx_df[["date", "spx_close"]].copy()
        spx["date"] = pd.to_datetime(spx["date"])
        spx = spx.sort_values("date").drop_duplicates("date")
        spx = spx[(spx["date"] >= start) & (spx["date"] <= end)]
        if len(spx) > 1:
            spx = spx.sort_values("date")
            ret = spx["spx_close"].pct_change().fillna(0)
            cum_ret = (1 + ret).cumprod()
            bench_pnl_m = (cum_ret.values - 1) * 1e6 / 1e6  # PnL in $M from $1M
            ax.plot(spx["date"], bench_pnl_m, label="Buy & Hold SPX", lw=1, color=PLOT_NEUTRAL, ls="--")
    ax.set_title("Cumulative PnL ($M) vs Buy & Hold SPX", fontsize=12, fontweight="bold")
    ax.set_ylabel("$ Millions")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}M"))
    ax.legend(fontsize=9)
    ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))

    # 2. Sharpe ratios (strategies only; benchmark in table)
    ax = axes[0, 1]
    sharpes = [r["sharpe"] for r in results if r["n_trades"] > 0]
    colors = [PLOT_POSITIVE if s > 0 else PLOT_NEGATIVE for s in sharpes]
    ax.bar(strategies, sharpes, color=colors, alpha=0.7, edgecolor="none")
    ax.set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
    ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Win Rate & # Trades
    ax = axes[0, 2]
    n_trades = [r["n_trades"] for r in results if r["n_trades"] > 0]
    win_rates = [r["win_rate"] * 100 for r in results if r["n_trades"] > 0]
    x = np.arange(len(strategies))
    ax.bar(x, n_trades, color=PLOT_SECONDARY, alpha=0.7, label="# Trades")
    ax2 = ax.twinx()
    ax2.plot(x, win_rates, "ro-", label="Win Rate %", ms=8)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_title("Trades & Win Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Trades")
    ax2.set_ylabel("Win Rate %")
    if win_rates:
        lo_wr = max(0, min(win_rates) - 10)
        hi_wr = min(100, max(win_rates) + 10)
        ax2.set_ylim(lo_wr, hi_wr)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Drawdown comparison (computed from daily_pnl cumsum to avoid cumprod blowup)
    ax = axes[1, 0]
    for name in strategies:
        if name in pnl_dict and not pnl_dict[name].empty:
            pdf = pnl_dict[name]
            cum_pnl_s = pdf["daily_pnl"].fillna(0).cumsum()
            notional_s = max(cum_pnl_s.abs().max(), 1)
            r_s = pdf["daily_pnl"].fillna(0) / notional_s
            cum_s = (1 + r_s).cumprod()
            dd = (cum_s - cum_s.cummax()) / cum_s.cummax()
            ax.plot(pdf["date"], dd, label=name, lw=0.8)
    ax.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
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

    # 6. Summary table (incl. PSR, Alpha vs benchmark)
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for r in results:
        if r["n_trades"] > 0:
            psr = r.get("psr", np.nan)
            alpha = r.get("alpha_ann", np.nan)
            psr_str = f"{psr:.0%}" if not (isinstance(psr, float) and np.isnan(psr)) else "—"
            alpha_str = "—" if (isinstance(alpha, float) and np.isnan(alpha)) else _fmt_alpha(alpha)
            table_data.append([
                r["strategy"],
                f"{r['n_trades']}",
                f"{r['win_rate']:.0%}",
                f"${r['total_pnl']/1e6:.2f}M",
                f"{r['sharpe']:.2f}",
                psr_str,
                alpha_str,
                f"{r['max_dd']:.1%}",
            ])
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["Strategy", "Trades", "Win%", "Total PnL", "Sharpe", "PSR(SR>0)", "Alpha", "Max DD"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(0.95, 1.8)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(PLOT_COLORS["600"])
                cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Performance Summary (PSR = Prob. Sharpe > 0)", fontsize=12, fontweight="bold")

    fig.suptitle("Strategy Comparison: VRP vs Variance Swap (vs Buy & Hold SPX)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "13_strategy_comparison")


def plot_fomc_analysis(fomc_results):
    """Compare FOMC vs non-FOMC: Sharpe, Total PnL, Win Rate, PSR(SR>0)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    strategies = list(fomc_results.keys())
    metrics = ["sharpe", "total_pnl", "win_rate", "psr"]
    titles = ["Sharpe Ratio", "Total PnL ($K)", "Win Rate (%)", "PSR (Prob. Sharpe > 0)"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        fomc_vals = []
        nonfomc_vals = []
        labels = []

        for strat in strategies:
            fr = fomc_results[strat]
            f_val = fr.get("fomc", {}).get(metric, 0)
            nf_val = fr.get("non_fomc", {}).get(metric, 0)
            if f_val is None or (isinstance(f_val, float) and np.isnan(f_val)):
                f_val = 0
            if nf_val is None or (isinstance(nf_val, float) and np.isnan(nf_val)):
                nf_val = 0

            if metric == "total_pnl":
                f_val /= 1000
                nf_val /= 1000
            elif metric == "win_rate":
                f_val *= 100
                nf_val *= 100
            elif metric == "psr":
                f_val = f_val * 100 if not np.isnan(f_val) else 0
                nf_val = nf_val * 100 if not np.isnan(nf_val) else 0

            fomc_vals.append(f_val)
            nonfomc_vals.append(nf_val)
            labels.append(strat)

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, fomc_vals, w, label="FOMC Window (±7d)",
               color=PLOT_ACCENT, alpha=0.8, edgecolor="none")
        ax.bar(x + w / 2, nonfomc_vals, w, label="Non-FOMC",
               color=PLOT_SECONDARY, alpha=0.8, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        if metric == "psr":
            ax.set_ylabel("%")

    fig.suptitle("FOMC Window (±7 days) vs Non-FOMC Performance (incl. PSR)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "14_fomc_analysis")


def main():
    print("\n" + "=" * 65)
    print("  STRATEGY COMPARISON — VRP vs Variance Swap")
    print("  + FOMC Window Analysis (±7 days)")
    print("=" * 65 + "\n")

    print("Loading data ...")
    data, augmented = load_data_and_augment()

    print("\n" + "─" * 50)
    print("  Building signals for each strategy ...")
    print("─" * 50)

    sig_vrp = _make_vrp_signal(augmented)
    sig_varswap = _make_varswap_signal(augmented)

    print("\n" + "─" * 50)
    print("  Backtesting each strategy ...")
    print("─" * 50)

    results = []
    pnl_dict = {}

    for name, sig_df in [
        ("VRP (ATM IV)", sig_vrp),
        ("Var Swap (MFIV)", sig_varswap),
    ]:
        if sig_df is None or sig_df.empty:
            results.append({"strategy": name, "n_trades": 0})
            continue

        print(f"\n  → {name}")
        trades, pnl_df = run_backtest(sig_df, data["spx"])
        results.append(_summarise(name, trades, pnl_df, spx_df=data["spx"]))
        pnl_dict[name] = pnl_df

    print("\n" + "─" * 50)
    print("  FOMC Window Analysis (±7 days) ...")
    print("─" * 50)

    fomc_results = {}
    for name, sig_df in [
        ("VRP (ATM IV)", sig_vrp),
        ("Var Swap (MFIV)", sig_varswap),
    ]:
        if sig_df is None or sig_df.empty:
            continue
        print(f"\n  → {name}")
        fomc_results[name] = _fomc_split(
            sig_df, data["spx"], augmented, name
        )

    print("\n" + "=" * 65)
    print("  STRATEGY COMPARISON RESULTS")
    print("=" * 65)

    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("  Strategy Comparison — VRP vs Variance Swap")
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
                f"    PSR (SR>0):    {r.get('psr', np.nan):.1%}",
                f"    Sortino:       {r['sortino']:.2f}",
                f"    Max Drawdown:  {r['max_dd']:.2%}",
                f"    Alpha (ann):   {_fmt_alpha(r.get('alpha_ann', np.nan))}",
                f"    Skewness:      {r['skewness']:.3f}",
                f"    Kurtosis:      {r['kurtosis']:.3f}",
            ]
            for l in lines:
                report_lines.append(l)
                print(l)
        report_lines.append("")

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

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "strategy_comparison.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved → {report_path}")

    print("\n  Generating plots ...")
    plot_strategy_comparison(results, pnl_dict, spx_df=data["spx"])
    if fomc_results:
        plot_fomc_analysis(fomc_results)

    print("\n" + "=" * 65)
    print("  COMPARISON COMPLETE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
