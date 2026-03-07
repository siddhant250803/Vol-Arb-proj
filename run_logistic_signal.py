#!/usr/bin/env python3
"""
Run the logistic-distribution RV forecast through the existing IV-RV signal
and backtest stack. Backtest holds only until option expiry (exdate from
feature table when present, else capped at 5 trading days for weeklies).
"""

from pathlib import Path

import pandas as pd

from src.config import DATA_OUTPUT_DIR, REPORTS_DIR, OPTIONS_DIR
from src.data_loader import load_options_raw, extract_spx_prices
from src.signals import compute_vrp_signal
from src.backtest import run_backtest, trades_to_dataframe
from src.performance import full_performance_report
from logistic import run_logistic_expected_vol_oos


def main() -> None:
    feature_candidates = [
        DATA_OUTPUT_DIR / "feature_table_stage1_stage2_notebook.csv",
        DATA_OUTPUT_DIR / "feature_table.csv",
    ]
    feature_path = next((path for path in feature_candidates if path.exists()), None)
    if feature_path is None:
        raise FileNotFoundError(
            "Missing feature table. Checked: "
            + ", ".join(str(path) for path in feature_candidates)
        )
    feature_df = pd.read_csv(feature_path, parse_dates=["date"])
    print(f"[logistic_signal] Using feature table: {feature_path}")

    options_candidates = [
        OPTIONS_DIR / "spx-weeklies_daily_friday-expiration_all.csv",
        OPTIONS_DIR / "spx-weeklies_daily_friday-expiration_all.csv.gz",
    ]
    options_path = next((path for path in options_candidates if path.exists()), None)
    if options_path is None:
        raise FileNotFoundError(
            "Could not find the options file for SPX extraction. Checked: "
            + ", ".join(str(path) for path in options_candidates)
        )
    raw_options = load_options_raw(filepath=options_path)
    spx_df = extract_spx_prices(raw_options)

    logistic_oos = run_logistic_expected_vol_oos(
        feature_df,
        return_col="log_return",
        n_quantiles=4,
        train_window=252,
    )

    feature_aug = feature_df.merge(
        logistic_oos[
            [
                "date",
                "expected_realised_volatility_plain",
                "expected_realised_volatility",
                "expected_realised_variance_plain",
                "expected_realised_variance",
            ]
        ],
        on="date",
        how="left",
    )
    feature_aug["logistic_rv_forecast"] = feature_aug["expected_realised_volatility_plain"]

    signal_df = compute_vrp_signal(
        feature_aug,
        iv_col="atm_iv_at_expiry",
        rv_col="logistic_rv_forecast",
    )

    trades, pnl_df = run_backtest(signal_df, spx_df)
    trades_df = trades_to_dataframe(trades)
    report = full_performance_report(pnl_df, trades_df, spx_df=spx_df)

    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logistic_oos.to_csv(DATA_OUTPUT_DIR / "logistic_oos_forecast.csv", index=False)
    signal_df.to_csv(DATA_OUTPUT_DIR / "logistic_signal_table.csv", index=False)
    trades_df.to_csv(DATA_OUTPUT_DIR / "logistic_trades.csv", index=False)
    pnl_df.to_csv(DATA_OUTPUT_DIR / "logistic_daily_pnl.csv", index=False)

    report_path = REPORTS_DIR / "logistic_signal_report.txt"
    with open(report_path, "w") as f:
        f.write("Logistic OOS IV-RV Signal Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Sharpe: {report['return_metrics']['sharpe_ratio']:.4f}\n")
        f.write(f"Annual Return: {report['return_metrics']['annualised_return']:.4%}\n")
        f.write(f"Annual Vol: {report['return_metrics']['annualised_volatility']:.4%}\n")
        f.write(f"Max Drawdown: {report['drawdown']['max_drawdown']:.4%}\n")
        f.write(f"Trades: {report['trade_stats']['n_trades']}\n")
        f.write(f"Win Rate: {report['trade_stats']['win_rate']:.4%}\n")
        f.write(f"Total PnL: {report['trade_stats']['total_pnl']:.2f}\n")

    print("\n[logistic_signal] Summary")
    print(f"Sharpe: {report['return_metrics']['sharpe_ratio']:.4f}")
    print(f"Annual Return: {report['return_metrics']['annualised_return']:.4%}")
    print(f"Annual Vol: {report['return_metrics']['annualised_volatility']:.4%}")
    print(f"Max Drawdown: {report['drawdown']['max_drawdown']:.4%}")
    print(f"Trades: {report['trade_stats']['n_trades']}")
    print(f"Total PnL: {report['trade_stats']['total_pnl']:.2f}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
