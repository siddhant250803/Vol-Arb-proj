#!/usr/bin/env python3
"""
Sweep logistic quantile counts and compare downstream strategy performance.
"""

from pathlib import Path

import pandas as pd

from src.config import DATA_OUTPUT_DIR, OPTIONS_DIR
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
    print(f"[logistic_sweep] Using feature table: {feature_path}")

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

    quantile_values = [3, 4, 5, 6, 8, 10]
    results = []

    for n_quantiles in quantile_values:
        print(f"\n[logistic_sweep] Running n_quantiles={n_quantiles}")
        logistic_oos = run_logistic_expected_vol_oos(
            feature_df,
            return_col="log_return",
            n_quantiles=n_quantiles,
            train_window=252,
        )

        feature_aug = feature_df.merge(
            logistic_oos[
                [
                    "date",
                    "expected_realised_volatility_plain",
                    "expected_realised_volatility",
                ]
            ],
            on="date",
            how="left",
        )
        feature_aug["logistic_rv_forecast"] = feature_aug["expected_realised_volatility_plain"]

        signal_df = compute_vrp_signal(
            feature_aug,
            iv_col="atm_iv_30d",
            rv_col="logistic_rv_forecast",
        )
        trades, pnl_df = run_backtest(signal_df, spx_df)
        trades_df = trades_to_dataframe(trades)
        report = full_performance_report(pnl_df, trades_df, spx_df=spx_df)

        rm = report["return_metrics"]
        ts = report["trade_stats"]
        results.append(
            {
                "n_quantiles": n_quantiles,
                "forecast_rows": len(logistic_oos),
                "signal_rows": len(signal_df),
                "sharpe": rm["sharpe_ratio"],
                "ann_return": rm["annualised_return"],
                "ann_vol": rm["annualised_volatility"],
                "max_drawdown": report["drawdown"]["max_drawdown"],
                "n_trades": ts["n_trades"],
                "win_rate": ts["win_rate"],
                "total_pnl": ts["total_pnl"],
            }
        )

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    out_path = DATA_OUTPUT_DIR / "logistic_quantile_sweep.csv"
    results_df.to_csv(out_path, index=False)

    print("\n[logistic_sweep] Results")
    print(results_df.to_string(index=False))
    print(f"\nSaved sweep results to {out_path}")


if __name__ == "__main__":
    main()
