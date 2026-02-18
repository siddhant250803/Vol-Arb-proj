#!/usr/bin/env python3
# ============================================================
# run_pipeline.py — End-to-End Execution of the Vol-Arb Strategy
# ============================================================
"""
Master script that runs every stage of the IV vs Forecast RV
volatility arbitrage pipeline:

    Stage 1 — Data Loading & Cleaning
    Stage 2 — Feature Engineering  (IV measures, RV, events)
    Stage 3 — RV Forecasting       (HAR-RV, GARCH family)
    Stage 4 — Signal Construction   (VRP, skew, term, distribution)
    Stage 5 — Backtesting           (delta-hedged straddles)
    Stage 6 — Performance Analysis  (metrics, robustness)
    Stage 7 — Visualisation         (all plots + dashboard)
    Stage 8 — Export Results        (CSV artifacts)

Usage:
    python run_pipeline.py              # full run
    python run_pipeline.py --quick      # fast dev run (fewer rows)
"""

import sys
import time
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

# ── Add project root to path ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OUTPUT_DIR, DATA_OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR,
)
from src.data_loader import load_all_data
from src.feature_engineering import build_feature_table
from src.rv_models import run_all_rv_models
from src.signals import build_signal_table
from src.backtest import run_backtest, trades_to_dataframe
from src.performance import (
    full_performance_report,
    robustness_by_subperiod,
    robustness_by_parameter,
)
from src import visualization as viz

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ════════════════════════════════════════════════════════════
#  PIPELINE STAGES
# ════════════════════════════════════════════════════════════

def stage1_load_data(quick=False):
    """
    Stage 1: Load and clean all raw data.

    Returns dict with keys: options, spx, yields, rf
    """
    print("\n" + "█" * 60)
    print("  STAGE 1 — DATA LOADING & CLEANING")
    print("█" * 60 + "\n")

    nrows = 50_000 if quick else None
    data = load_all_data(nrows=nrows)

    print(f"\n  Options: {len(data['options']):,} rows")
    print(f"  SPX prices: {len(data['spx']):,} trading days")
    print(f"  Yield curve: {len(data['yields']):,} days")
    return data


def stage2_features(data):
    """
    Stage 2: Engineer all features (IV, RV, events).

    Returns the master feature table.
    """
    print("\n" + "█" * 60)
    print("  STAGE 2 — FEATURE ENGINEERING")
    print("█" * 60 + "\n")

    features = build_feature_table(
        data["options"],
        data["spx"],
        rf_series=data["rf"],
    )
    return features


def stage3_rv_forecasts(features):
    """
    Stage 3: Run all RV forecast models and merge with features.

    Returns augmented feature table.
    """
    print("\n" + "█" * 60)
    print("  STAGE 3 — RV FORECASTING MODELS")
    print("█" * 60 + "\n")

    forecasts = run_all_rv_models(features, train_window=252)

    # Merge forecasts back into feature table
    augmented = features.merge(forecasts, on="date", how="left")
    print(f"\n  Augmented feature table: {len(augmented)} rows, "
          f"{augmented.shape[1]} columns")
    return augmented


def stage4_signals(augmented, data):
    """
    Stage 4: Construct all trading signals.

    Returns signal table.
    """
    print("\n" + "█" * 60)
    print("  STAGE 4 — SIGNAL CONSTRUCTION")
    print("█" * 60 + "\n")

    signals = build_signal_table(augmented, data["options"], data["spx"])
    return signals


def stage5_backtest(signals, data):
    """
    Stage 5: Run the backtest.

    Returns (trades, daily_pnl).
    """
    print("\n" + "█" * 60)
    print("  STAGE 5 — BACKTESTING")
    print("█" * 60 + "\n")

    trades, pnl_df = run_backtest(signals, data["spx"])
    trades_df = trades_to_dataframe(trades)
    return trades, trades_df, pnl_df


def stage6_performance(trades_df, pnl_df, signals, data):
    """
    Stage 6: Compute performance metrics and robustness checks.

    Returns (report, subperiod_df, param_df).
    """
    print("\n" + "█" * 60)
    print("  STAGE 6 — PERFORMANCE ANALYSIS")
    print("█" * 60 + "\n")

    report = {}
    subperiod_df = pd.DataFrame()
    param_df = pd.DataFrame()

    if not pnl_df.empty and len(pnl_df) > 10:
        report = full_performance_report(pnl_df, trades_df)

        # Sub-period analysis
        n_periods = min(4, max(2, len(pnl_df) // 50))
        subperiod_df = robustness_by_subperiod(pnl_df, n_periods=n_periods)
        print("\n  Sub-period Sharpe ratios:")
        print(subperiod_df[["period", "start", "end", "sharpe", "ann_return"]].to_string())

        # Parameter sensitivity (quick grid)
        print("\n  Running parameter sensitivity ...")
        param_df = robustness_by_parameter(
            signals, data["spx"], run_backtest,
            hold_days_range=[15, 22, 30],
            cost_range=[0, 5, 10],
        )
        if not param_df.empty:
            print(param_df.to_string())
    else:
        print("  [!] Not enough PnL data for performance analysis.")

    return report, subperiod_df, param_df


def stage7_visualise(data, features, signals, pnl_df, trades_df,
                     report, subperiod_df, param_df):
    """
    Stage 7: Generate all visualisations.
    """
    print("\n" + "█" * 60)
    print("  STAGE 7 — VISUALISATION")
    print("█" * 60 + "\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # A. Data exploration
    viz.plot_spx_price_and_returns(data["spx"])
    viz.plot_options_summary(data["options"])

    # B. Features & signals
    viz.plot_iv_vs_rv(features)
    viz.plot_rv_forecasts(features)
    viz.plot_vrp_signal(signals)
    viz.plot_skew_and_term_signals(signals)

    # C. Backtest performance
    if not pnl_df.empty:
        viz.plot_cumulative_pnl(pnl_df)
        viz.plot_trade_analysis(trades_df)
        viz.plot_monthly_returns(pnl_df)

    # D. Robustness
    if not subperiod_df.empty:
        viz.plot_robustness_subperiods(subperiod_df)
    if not param_df.empty:
        viz.plot_parameter_sensitivity(param_df)

    # E. Dashboard
    if report:
        viz.plot_summary_dashboard(data["spx"], features, signals, pnl_df, report)

    print(f"\n  All figures saved to: {FIGURES_DIR}")


def stage8_export(data, features, signals, trades_df, pnl_df, report):
    """
    Stage 8: Export processed data and results to CSV.
    """
    print("\n" + "█" * 60)
    print("  STAGE 8 — EXPORTING RESULTS")
    print("█" * 60 + "\n")

    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Feature table
    features.to_csv(DATA_OUTPUT_DIR / "feature_table.csv", index=False)
    print(f"  Saved feature_table.csv ({len(features)} rows)")

    # Signals
    signals.to_csv(DATA_OUTPUT_DIR / "signal_table.csv", index=False)
    print(f"  Saved signal_table.csv ({len(signals)} rows)")

    # Trades
    if not trades_df.empty:
        trades_df.to_csv(DATA_OUTPUT_DIR / "trades.csv", index=False)
        print(f"  Saved trades.csv ({len(trades_df)} trades)")

    # Daily PnL
    if not pnl_df.empty:
        pnl_df.to_csv(DATA_OUTPUT_DIR / "daily_pnl.csv", index=False)
        print(f"  Saved daily_pnl.csv ({len(pnl_df)} days)")

    # Performance report (as text)
    if report:
        with open(REPORTS_DIR / "performance_report.txt", "w") as f:
            f.write("=" * 60 + "\n")
            f.write("  IV vs Forecast RV — Performance Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("RETURN METRICS\n")
            for k, v in report.get("return_metrics", {}).items():
                f.write(f"  {k:30s}: {v:.4f}\n")

            f.write("\nDRAWDOWN\n")
            dd = report.get("drawdown", {})
            f.write(f"  {'max_drawdown':30s}: {dd.get('max_drawdown', 0):.4f}\n")

            f.write("\nDISTRIBUTION\n")
            for k, v in report.get("distribution", {}).items():
                f.write(f"  {k:30s}: {v:.6f}\n")

            f.write("\nTRADE STATISTICS\n")
            for k, v in report.get("trade_stats", {}).items():
                if isinstance(v, float):
                    f.write(f"  {k:30s}: {v:.4f}\n")
                else:
                    f.write(f"  {k:30s}: {v}\n")

        print(f"  Saved performance_report.txt")

    print(f"\n  All outputs in: {OUTPUT_DIR}")


# ════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run the IV vs Forecast RV volatility arbitrage pipeline."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast development run (fewer rows, smaller windows)."
    )
    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "=" * 60)
    print("  IV vs FORECAST RV — VOLATILITY ARBITRAGE PIPELINE")
    print("  MS&E 244, Stanford University")
    print("=" * 60)
    if args.quick:
        print("  ⚡ QUICK MODE — reduced data for development")
    print()

    # ── Run all stages ─────────────────────────────────────
    data = stage1_load_data(quick=args.quick)
    features = stage2_features(data)
    augmented = stage3_rv_forecasts(features)
    signals = stage4_signals(augmented, data)
    trades, trades_df, pnl_df = stage5_backtest(signals, data)
    report, subperiod_df, param_df = stage6_performance(
        trades_df, pnl_df, signals, data
    )
    stage7_visualise(data, augmented, signals, pnl_df, trades_df,
                     report, subperiod_df, param_df)
    stage8_export(data, augmented, signals, trades_df, pnl_df, report)

    # ── Done ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE — {elapsed:.1f}s elapsed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
