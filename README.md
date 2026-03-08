# IV vs Forecast RV — Volatility Arbitrage Strategy

**MS&E 244 · Stanford University · Group 4**

A systematic volatility-trading strategy that tests whether SPX options implied volatility (IV) is systematically richer than subsequently realised volatility (RV), and whether the IV–RV spread (variance risk premium) is tradeable after realistic costs and hedging frictions.

---

## Research Objective

1. **Variance Risk Premium (VRP):** Risk-neutral implied variance typically exceeds physical realised variance. We quantify this premium and test its predictive power.
2. **Tradeable Signal:** Construct a z-scored VRP signal and trade delta-hedged ATM straddles when the signal is extreme.
3. **Strategies Compared:** VRP (ATM IV), Var Swap (MFIV), and Logistic regression RV forecast.

---

## Project Structure

```
Vol-Arb-proj/
├── src/                          # Core Python modules
│   ├── config.py                 # Paths, constants, parameters
│   ├── data_loader.py            # Load & clean options, yields, SPX
│   ├── feature_engineering.py    # ATM IV, MFIV, realised vol (at expiry)
│   ├── rv_models.py              # HAR-RV, GARCH, GJR-GARCH
│   ├── signals.py                # VRP, Var Swap signals
│   ├── backtest.py               # Delta-hedged straddle backtester
│   ├── performance.py            # Sharpe, Sortino, drawdowns, robustness
│   └── visualization.py         # Publication-quality plots
│
├── run_pipeline.py               # End-to-end pipeline
├── run_comparison.py             # VRP vs Var Swap comparison
├── run_robustness.py             # OOS, stress, regime, param sensitivity
├── run_logistic_signal.py        # Logistic RV forecast backtest
├── run_logistic_quantile_sweep.py # Logistic quantile sweep (K=3..10)
├── logistic.py                   # Logistic regression RV forecast
├── requirements.txt
│
├── Group 4 MS&E244/              # Raw data (not committed)
│   ├── Options/
│   └── Risk-Free/
│
├── output/                       # Generated outputs
│   ├── figures/                  # PNG charts (01–20)
│   ├── data/                     # feature_table, signal_table, trades, etc.
│   └── reports/                  # performance_report, strategy_comparison, robustness_report
│
└── report/
    ├── report.tex                # Full academic report
    ├── presentation.tex         # Beamer slides
    └── pipeline_flow.tex        # Pipeline flow diagram
```

---

## Quick Start

### 1. Create & activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
# Full run (figures 01–12)
python run_pipeline.py

# Quick dev run (50K rows)
python run_pipeline.py --quick

# Full run + comparison, robustness, logistic sweep (figures 01–20)
python run_pipeline.py --all-figures
```

### 4. Run individual scripts

```bash
python run_comparison.py       # VRP vs Var Swap
python run_robustness.py       # OOS, stress, regime, param sensitivity
python run_logistic_signal.py  # Logistic RV (K=4)
python run_logistic_quantile_sweep.py  # Logistic quantile sweep
```

### 5. Build report and presentation

```bash
cd report
pdflatex report.tex
pdflatex presentation.tex
```

---

## Key Results (2011–2023, 25% stop per trade)

| Strategy | Trades | Sharpe | Total PnL | Max DD |
|----------|--------|--------|-----------|--------|
| VRP (ATM IV, composite RV) | 188 | 2.16 | $16.54M | −83.9% |
| Var Swap (MFIV) | 190 | 2.54 | $24.89M | −82.3% |
| Logistic (K=8) | 186 | 2.11 | $16.62M | −84.9% |

- **Var Swap** outperforms VRP on Sharpe and PnL.
- **25% stop per trade** (early exit and expiry) caps losses; max DD ~84%.
- Profitable in **11 of 13 years**; strong in stress (Volmageddon, COVID); profitable across regimes.
- **Bootstrap:** P(Sharpe > 0) = 99.9%; 95% CI [0.86, 10.31].

---

## Pipeline Stages

1. **Data Loading** — OptionMetrics SPX weeklies, Treasury yields, filters (bid, spread, DTE, moneyness).
2. **Feature Engineering** — ATM IV, MFIV, RV at expiry; FOMC flags.
3. **RV Forecasting** — HAR-RV, GARCH, GJR; equal-weight composite.
4. **Signal Construction** — VRP z-score; enter when |z| > 1.
5. **Backtesting** — Delta-hedged straddles, hold to expiry (1–5d), 25% stop per trade, 5 bps cost.
6. **Performance** — Sharpe, Sortino, drawdown, trade stats.
7. **Visualisation** — 20 figures (SPX, IV/RV, signals, PnL, robustness, etc.).
8. **Export** — CSVs and reports.

---

## Outputs

| Path | Contents |
|------|----------|
| `output/figures/*.png` | 20 charts |
| `output/data/feature_table.csv` | Master feature table |
| `output/data/signal_table.csv` | Trading signals |
| `output/data/trades.csv` | Trade records |
| `output/data/daily_pnl.csv` | Daily PnL |
| `output/reports/performance_report.txt` | Full performance |
| `output/reports/strategy_comparison.txt` | VRP vs Var Swap |
| `output/reports/robustness_report.txt` | OOS, stress, regime, params |

---

## License

Academic project — MS&E 244, Stanford University.
