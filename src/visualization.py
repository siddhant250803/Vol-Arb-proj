"""
Plotting utilities for data exploration, signals, backtest performance,
robustness analysis, and summary dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import (
    FIGURES_DIR, FIGURE_DPI, FIGURE_SIZE, STYLE,
    PLOT_COLORS, PLOT_PALETTE, PLOT_PRIMARY, PLOT_SECONDARY,
    PLOT_LIGHT, PLOT_ACCENT, PLOT_NEUTRAL, PLOT_POSITIVE, PLOT_NEGATIVE,
)

try:
    plt.style.use(STYLE)
except OSError:
    plt.style.use("seaborn-v0_8")
sns.set_palette(PLOT_PALETTE)


def _save(fig, name):
    """Save a figure to the output directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  [viz] Saved → {path.relative_to(FIGURES_DIR.parent.parent)}")
    return fig


def plot_spx_price_and_returns(spx_df):
    """
    Two-panel chart:
        Top: SPX price level
        Bottom: Daily log-returns
    """
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax1, ax2 = axes
    ax1.plot(spx_df["date"], spx_df["spx_close"], lw=0.8, color=PLOT_SECONDARY)
    ax1.set_ylabel("SPX Price")
    ax1.set_title("SPX Price Level & Daily Returns", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.bar(spx_df["date"], spx_df["log_return"], width=1, color=PLOT_NEUTRAL, alpha=0.6)
    ax2.set_ylabel("Log Return")
    ax2.set_xlabel("Date")
    ax2.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "01_spx_price_returns")


def plot_options_summary(options_df):
    """
    Multi-panel summary of the options data:
        1. Option volume over time
        2. IV distribution
        3. Moneyness distribution
        4. DTE distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    vol_ts = options_df.groupby("date")["volume"].sum()
    axes[0, 0].plot(vol_ts.index, vol_ts.values, lw=0.5, color=PLOT_SECONDARY)
    axes[0, 0].set_title("Total Daily Option Volume")
    axes[0, 0].set_ylabel("Contracts")

    axes[0, 1].hist(options_df["impl_volatility"].dropna(), bins=100,
                    color=PLOT_ACCENT, alpha=0.7, edgecolor="none")
    axes[0, 1].set_title("Implied Volatility Distribution")
    axes[0, 1].set_xlabel("IV")

    axes[1, 0].hist(options_df["moneyness"], bins=100,
                    color=PLOT_COLORS["300"], alpha=0.7, edgecolor="none")
    axes[1, 0].set_title("Strike Moneyness (K/S)")
    axes[1, 0].axvline(1.0, color=PLOT_PRIMARY, ls="--", lw=1)

    axes[1, 1].hist(options_df["dte"], bins=60,
                    color=PLOT_COLORS["600"], alpha=0.7, edgecolor="none")
    axes[1, 1].set_title("Days to Expiration")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("Options Data Summary", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "02_options_summary")


def plot_iv_vs_rv(feature_df):
    """
    Time-series overlay of ATM IV (30d) vs realised volatility.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    df = feature_df.dropna(subset=["atm_iv_at_expiry", "rvol_monthly"])

    ax.plot(df["date"], df["atm_iv_at_expiry"], label="ATM IV (at expiry)", lw=1.2, color=PLOT_COLORS["700"])
    ax.plot(df["date"], df["rvol_monthly"], label="Realised Vol (22d)", lw=1.2, color=PLOT_PRIMARY)
    ax.fill_between(
        df["date"],
        df["atm_iv_at_expiry"],
        df["rvol_monthly"],
        alpha=0.15,
        color=PLOT_NEUTRAL,
        label="VRP Spread",
    )
    ax.set_ylabel("Annualised Volatility")
    ax.set_title("Implied Volatility vs Realised Volatility", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "03_iv_vs_rv")


def plot_rv_forecasts(feature_df):
    """
    Compare realised RV with model forecasts (HAR, GARCH, etc.).
    """
    fcast_cols = [c for c in feature_df.columns if "forecast" in c]
    if not fcast_cols:
        print("  [viz] No forecast columns found — skipping.")
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if "fwd_rvol" in feature_df.columns:
        ax.plot(feature_df["date"], feature_df["fwd_rvol"],
                label="Actual Forward RV", lw=1.0, color=PLOT_NEUTRAL, alpha=0.6)

    colors = PLOT_PALETTE
    for i, col in enumerate(fcast_cols):
        numeric_col = pd.to_numeric(feature_df[col], errors="coerce").clip(lower=0)
        vals = np.sqrt(numeric_col.astype(float))
        ax.plot(feature_df["date"], vals,
                label=col.replace("_", " ").title(),
                lw=0.8, color=colors[i % len(colors)], alpha=0.7)

    ax.set_ylabel("Annualised Volatility")
    ax.set_title("RV Forecasts vs Actual", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "04_rv_forecasts")


def plot_vrp_signal(signal_df):
    """
    Three-panel VRP signal plot:
        1. VRP level
        2. VRP z-score with entry thresholds
        3. Signal direction (+1, 0, −1)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2, 0.6]})

    # VRP level
    axes[0].plot(signal_df["date"], signal_df["vrp"], lw=0.8, color=PLOT_SECONDARY)
    axes[0].axhline(0, color=PLOT_NEUTRAL, lw=0.5, ls="--")
    axes[0].set_ylabel("VRP")
    axes[0].set_title("Variance Risk Premium Signal", fontsize=14, fontweight="bold")

    # Z-score
    axes[1].plot(signal_df["date"], signal_df["vrp_zscore"], lw=0.8, color=PLOT_COLORS["800"])
    axes[1].axhline(1.0, color=PLOT_PRIMARY, ls="--", lw=0.8, label="Entry threshold")
    axes[1].axhline(-1.0, color=PLOT_POSITIVE, ls="--", lw=0.8)
    axes[1].axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    axes[1].fill_between(signal_df["date"], 1.0, signal_df["vrp_zscore"],
                         where=signal_df["vrp_zscore"] > 1.0,
                         alpha=0.3, color=PLOT_PRIMARY, label="Short vol zone")
    axes[1].fill_between(signal_df["date"], -1.0, signal_df["vrp_zscore"],
                         where=signal_df["vrp_zscore"] < -1.0,
                         alpha=0.3, color=PLOT_POSITIVE, label="Long vol zone")
    axes[1].set_ylabel("Z-Score")
    axes[1].legend(loc="upper right", fontsize=9)

    # Signal
    axes[2].bar(signal_df["date"], signal_df["signal"], width=2,
                color=signal_df["signal"].map({1: PLOT_PRIMARY, -1: PLOT_POSITIVE, 0: PLOT_NEUTRAL}))
    axes[2].set_ylabel("Signal")
    axes[2].set_yticks([-1, 0, 1])
    axes[2].set_yticklabels(["Long Vol", "Flat", "Short Vol"])

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "05_vrp_signal")


def plot_skew_and_term_signals(signal_df):
    """Plot skew and term-structure signals if available."""
    cols = [c for c in ["skew_signal", "term_signal"]
            if c in signal_df.columns]
    if not cols:
        return None

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 4 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    titles = {
        "skew_signal": "Skew Mispricing Signal",
        "term_signal": "Term-Structure Signal (Short − Long IV)",
    }
    colors = {k: v for k, v in zip(["skew_signal", "term_signal"], PLOT_PALETTE[:2])}

    for ax, col in zip(axes, cols):
        data = signal_df.dropna(subset=[col])
        ax.plot(data["date"], data[col], lw=0.8, color=colors.get(col, PLOT_NEUTRAL))
        ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5, ls="--")
        ax.set_ylabel(col)
        ax.set_title(titles.get(col, col), fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "06_surface_signals")


def plot_cumulative_pnl(pnl_df):
    """Cumulative PnL and drawdown chart."""
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(pnl_df["date"], pnl_df["cumulative_pnl"],
                 lw=1.2, color=PLOT_COLORS["800"])
    axes[0].fill_between(pnl_df["date"], 0, pnl_df["cumulative_pnl"],
                         alpha=0.1, color=PLOT_PRIMARY)
    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[0].set_title("Strategy Cumulative PnL & Drawdown",
                      fontsize=14, fontweight="bold")
    axes[0].axhline(0, color=PLOT_NEUTRAL, lw=0.5)

    r = pnl_df["daily_return"].fillna(0)
    cum_ret = (1 + r).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    axes[1].fill_between(pnl_df["date"], dd, 0, color=PLOT_ACCENT, alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "07_cumulative_pnl")


def plot_trade_analysis(trades_df):
    """Four-panel trade analysis."""
    if trades_df.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].hist(trades_df["net_pnl"], bins=30, color=PLOT_SECONDARY,
                    alpha=0.7, edgecolor="none")
    axes[0, 0].axvline(0, color=PLOT_PRIMARY, ls="--")
    axes[0, 0].set_title("Trade PnL Distribution")
    axes[0, 0].set_xlabel("Net PnL ($)")

    for direction, color, label in [(-1, PLOT_POSITIVE, "Short Vol"), (1, PLOT_PRIMARY, "Long Vol")]:
        sub = trades_df[trades_df["direction"] == direction]
        if len(sub) > 0:
            axes[0, 1].hist(sub["net_pnl"], bins=20, color=color,
                           alpha=0.5, label=label, edgecolor="none")
    axes[0, 1].legend()
    axes[0, 1].set_title("PnL by Direction")
    axes[0, 1].axvline(0, color=PLOT_NEUTRAL, ls="--")

    axes[1, 0].scatter(trades_df["iv_rv_spread"], trades_df["net_pnl"],
                       alpha=0.5, s=15, c=trades_df["direction"],
                       cmap="RdYlGn")
    axes[1, 0].axhline(0, color=PLOT_NEUTRAL, ls="--")
    axes[1, 0].axvline(0, color=PLOT_NEUTRAL, ls="--")
    axes[1, 0].set_xlabel("IV − RV Spread at Entry")
    axes[1, 0].set_ylabel("Net PnL ($)")
    axes[1, 0].set_title("Entry Spread vs Trade PnL")

    cum_pnl = trades_df["net_pnl"].cumsum()
    axes[1, 1].plot(range(len(cum_pnl)), cum_pnl, color=PLOT_COLORS["800"], lw=1.2)
    axes[1, 1].fill_between(range(len(cum_pnl)), 0, cum_pnl, alpha=0.1, color=PLOT_PRIMARY)
    axes[1, 1].set_xlabel("Trade Number")
    axes[1, 1].set_ylabel("Cumulative PnL ($)")
    axes[1, 1].set_title("Cumulative PnL Over Trades")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("Trade-Level Analysis", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "08_trade_analysis")


def plot_monthly_returns(pnl_df):
    """Monthly returns heatmap."""
    df = pnl_df[["date", "daily_return"]].copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = df.groupby(["year", "month"])["daily_return"].sum().unstack()

    fig, ax = plt.subplots(figsize=(14, max(6, len(monthly) * 0.5)))
    sns.heatmap(
        monthly,
        annot=True,
        fmt=".2%",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    fig.tight_layout()
    return _save(fig, "09_monthly_returns")


def plot_robustness_subperiods(subperiod_df):
    """Bar chart of Sharpe ratio by sub-period."""
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f"P{r['period']}\n{r['start'].strftime('%Y-%m')} → "
              f"{r['end'].strftime('%Y-%m')}"
              for _, r in subperiod_df.iterrows()]
    colors = [PLOT_POSITIVE if s > 0 else PLOT_NEGATIVE for s in subperiod_df["sharpe"]]

    ax.bar(labels, subperiod_df["sharpe"], color=colors, alpha=0.7, edgecolor="none")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Sub-Period", fontsize=14, fontweight="bold")
    ax.axhline(0, color=PLOT_NEUTRAL, lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return _save(fig, "10_robustness_subperiods")


def plot_parameter_sensitivity(param_df):
    """Heatmap of Sharpe across holding period × cost grid. Hold days ≤ expiry (weeklies: max 5d)."""
    if param_df.empty:
        return None

    pivot = param_df.pivot_table(
        index="hold_days", columns="cost_bps", values="sharpe"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Sharpe Ratio: Hold Days (≤ expiry) × Transaction Cost",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Hold days (≤ option expiry)")

    fig.tight_layout()
    return _save(fig, "11_parameter_sensitivity")


def plot_summary_dashboard(spx_df, feature_df, signal_df, pnl_df, report):
    """
    Single-page 6-panel summary dashboard.
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(spx_df["date"], spx_df["spx_close"], lw=0.7, color=PLOT_SECONDARY)
    ax1.set_title("SPX Price", fontsize=11, fontweight="bold")
    ax1.tick_params(labelsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    df2 = feature_df.dropna(subset=["atm_iv_at_expiry", "rvol_monthly"])
    if len(df2) > 0:
        ax2.plot(df2["date"], df2["atm_iv_at_expiry"], lw=0.7, label="IV", color=PLOT_COLORS["700"])
        ax2.plot(df2["date"], df2["rvol_monthly"], lw=0.7, label="RV", color=PLOT_PRIMARY)
        ax2.legend(fontsize=8)
    ax2.set_title("IV vs RV", fontsize=11, fontweight="bold")
    ax2.tick_params(labelsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(signal_df["date"], signal_df["vrp_zscore"], lw=0.7, color=PLOT_COLORS["800"])
    ax3.axhline(1, color=PLOT_PRIMARY, ls="--", lw=0.5)
    ax3.axhline(-1, color=PLOT_POSITIVE, ls="--", lw=0.5)
    ax3.set_title("VRP Z-Score", fontsize=11, fontweight="bold")
    ax3.tick_params(labelsize=8)

    ax4 = fig.add_subplot(gs[1, :2])
    if not pnl_df.empty:
        ax4.plot(pnl_df["date"], pnl_df["cumulative_pnl"], lw=1.0, color=PLOT_COLORS["800"])
        ax4.fill_between(pnl_df["date"], 0, pnl_df["cumulative_pnl"],
                        alpha=0.1, color=PLOT_PRIMARY)
    ax4.set_title("Cumulative PnL", fontsize=11, fontweight="bold")
    ax4.tick_params(labelsize=8)
    ax4.axhline(0, color=PLOT_NEUTRAL, lw=0.5)

    # Panel 5: Performance metrics text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    rm = report["return_metrics"]
    ts = report["trade_stats"]
    text = (
        f"Sharpe:    {rm['sharpe_ratio']:.2f}\n"
        f"Sortino:   {rm['sortino_ratio']:.2f}\n"
        f"Ann. Ret:  {rm['annualised_return']:.1%}\n"
        f"Ann. Vol:  {rm['annualised_volatility']:.1%}\n"
        f"Max DD:    {report['drawdown']['max_drawdown']:.1%}\n"
        f"─────────────\n"
        f"Trades:    {ts.get('n_trades', 0)}\n"
        f"Win Rate:  {ts.get('win_rate', 0):.1%}\n"
        f"Avg PnL:   ${ts.get('avg_pnl', 0):,.0f}\n"
        f"Total PnL: ${ts.get('total_pnl', 0):,.0f}\n"
    )
    ax5.text(0.1, 0.95, text, transform=ax5.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor=PLOT_LIGHT, alpha=0.7))
    ax5.set_title("Key Metrics", fontsize=11, fontweight="bold")

    # Panel 6: Drawdown
    ax6 = fig.add_subplot(gs[2, :])
    if not pnl_df.empty:
        r = pnl_df["daily_return"].fillna(0)
        cum_ret = (1 + r).cumprod()
        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        ax6.fill_between(pnl_df["date"], dd, 0, color=PLOT_ACCENT, alpha=0.4)
    ax6.set_title("Drawdown", fontsize=11, fontweight="bold")
    ax6.tick_params(labelsize=8)

    for ax in [ax1, ax2, ax3, ax4, ax6]:
        ax.grid(True, alpha=0.2)

    fig.suptitle("IV vs Forecast RV — Volatility Arbitrage Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)

    return _save(fig, "12_summary_dashboard")
