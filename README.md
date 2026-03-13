# IV vs Forecast RV — Volatility Arbitrage Strategy

**MS&E 244 · Stanford University · Group 4**

A systematic volatility-trading strategy that exploits the variance risk premium (VRP) in SPX weekly options. The strategy sells delta-hedged ATM straddles when implied volatility is rich relative to forecast realised volatility, and buys straddles when IV is cheap.

## Project Structure

```
Vol-Arb-proj/
├── src/                             Core library
│   ├── config.py                    Paths, constants, parameters
│   ├── data_loader.py               Load & clean OptionMetrics data
│   ├── feature_engineering.py       ATM IV, MFIV, realised vol, event flags
│   ├── rv_models.py                 HAR-RV, GARCH, GJR-GARCH forecasts
│   ├── signals.py                   VRP, skew, term-structure signals
│   ├── backtest.py                  Delta-hedged straddle backtester
│   ├── performance.py               Metrics, benchmark comparison, robustness
│   └── visualization.py             20 publication-quality figures
│
├── logistic.py                      Logistic quantile-bucket RV forecast
├── run_pipeline.py                  End-to-end pipeline
├── run_comparison.py                VRP vs Var Swap + FOMC analysis
├── run_robustness.py                OOS, walk-forward, stress, bootstrap
├── run_logistic_signal.py           Logistic RV forecast backtest
├── run_logistic_quantile_sweep.py   Quantile sweep (K = 3..10)
├── requirements.txt
│
├── Group 4 MS&E244/                 Raw data (not committed)
│   ├── Options/                     OptionMetrics SPX weeklies
│   └── Risk-Free/                   Zero-coupon yield panel
│
├── output/                          Generated outputs (not committed)
│   ├── figures/                     Figures 01–20
│   ├── data/                        Feature table, signals, trades, PnL
│   └── reports/                     Performance and robustness reports
│
└── report/
    ├── report.tex                   Academic report
    └── presentation.tex             Beamer slides
```

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (figures 01–12)
python run_pipeline.py

# 4. Run everything including robustness and comparison (figures 01–20)
python run_pipeline.py --all-figures

# 5. Or run individual analyses
python run_comparison.py                # VRP vs Var Swap (figures 13–14)
python run_robustness.py                # OOS, stress, regime (figures 15–20)
python run_logistic_signal.py           # Logistic RV forecast backtest
python run_logistic_quantile_sweep.py   # Quantile sweep
```

## Pipeline Stages

1. **Data Loading** — OptionMetrics SPX weeklies (Friday expiration), Treasury yields. Filters: bid > $0.05, spread ratio < 100%, DTE 5–9 (entry-day), moneyness ±20%.
2. **Feature Engineering** — ATM IV (interpolated to expiry), MFIV (Carr-Madan variance swap strike), rolling RV (daily/weekly/monthly), bipower variation, forward RV at expiry, FOMC flags.
3. **RV Forecasting** — HAR-RV (Corsi 2009), GARCH(1,1), GJR-GARCH; equal-weight composite forecast. All expanding-window out-of-sample.
4. **Signal Construction** — VRP z-score = (IV − forecast RV − rolling mean) / rolling std. Enter when |z| > 1.
5. **Backtesting** — Delta-hedged ATM straddles, $1M notional per trade. Hold to option expiry (max 5 trading days). 25% per-trade stop-loss. 5 bps transaction cost.
6. **Performance** — Sharpe, Sortino, Calmar, PSR, drawdown, benchmark comparison (vs buy-and-hold SPX), trade statistics.
7. **Visualisation** — 20 figures: SPX prices, options summary, IV/RV, forecasts, signals, cumulative PnL, trade analysis, monthly returns, robustness, parameter sensitivity, bootstrap, yearly breakdown.

## Key Results (2011–2023)

| Strategy | Trades | Sharpe | Total PnL | Max DD |
|---|---|---|---|---|
| VRP (ATM IV, composite RV) | 128 | 0.49 | $15.5M | −80.1% |
| Var Swap (MFIV) | 113 | 0.81 | $22.0M | −76.2% |

- **Var Swap** outperforms VRP on Sharpe and total PnL.
- 25% per-trade stop-loss caps individual losses; max drawdown ~78%.
- **PSR** (probability that true Sharpe > 0) = 100% for both strategies.
- Profitable across volatility regimes and most stress periods.
- Bootstrap 95% CI for Sharpe well above zero.

## Outputs

| Path | Contents |
|---|---|
| `output/figures/*.png` | 20 charts (01–20) |
| `output/data/feature_table.csv` | Master feature table |
| `output/data/signal_table.csv` | Trading signals |
| `output/data/trades.csv` | Trade records |
| `output/data/daily_pnl.csv` | Daily PnL |
| `output/reports/performance_report.txt` | Full performance metrics |
| `output/reports/strategy_comparison.txt` | VRP vs Var Swap comparison |
| `output/reports/robustness_report.txt` | OOS, stress, regime, bootstrap |

## License

Academic project — MS&E 244, Stanford University.
