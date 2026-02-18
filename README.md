# IV vs Forecast RV — Volatility Arbitrage Strategy

**MS&E 244 · Stanford University · Group 4**

A systematic volatility-trading strategy that tests whether SPX options implied
volatility (IV) is systematically richer than subsequently realised volatility
(RV), and whether the IV-RV spread (variance risk premium) is tradeable after
realistic costs and hedging frictions.

---

## Research Objective

1. **Variance Risk Premium (VRP):** Risk-neutral implied variance typically
   exceeds physical realised variance. We quantify this premium and test its
   predictive power.
2. **Tradeable Signal:** Construct a z-scored VRP signal and trade delta-hedged
   ATM straddles when the signal is extreme.
3. **Surface Mispricing:** Explore skew and term-structure signals as secondary
   alpha sources.

---

## Project Structure

```
Vol-Arb-proj/
├── src/                          # Core Python modules
│   ├── __init__.py               # Package docstring
│   ├── config.py                 # All paths, constants, parameters
│   ├── data_loader.py            # Load & clean options, yields, SPX
│   ├── feature_engineering.py    # ATM IV, model-free IV, realised vol
│   ├── rv_models.py              # HAR-RV, GARCH, EGARCH, GJR-GARCH
│   ├── signals.py                # VRP, skew, term-structure, distribution
│   ├── backtest.py               # Delta-hedged straddle backtester
│   ├── performance.py            # Sharpe, Sortino, drawdowns, robustness
│   └── visualization.py          # 12+ publication-quality plots
│
├── run_pipeline.py               # One-command end-to-end execution
├── requirements.txt              # Python dependencies
├── proposal.md                   # Original research proposal
│
├── Group 4 MS&E244/              # Raw data (not committed)
│   ├── Options/
│   │   ├── spx-weeklies-filtered.csv
│   │   ├── spx-weeklies-all.csv
│   │   └── options-data-dictionary.csv
│   └── Risk-Free/
│       └── yield_panel_daily_frequency_monthly_maturity.csv
│
└── output/                       # Generated outputs
    ├── figures/                   # All visualisations (PNG)
    ├── data/                      # Processed CSVs
    └── reports/                   # Performance report (TXT)
```

---

## Quick Start

### 1. Create & activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
# Full run (all data, all models)
python run_pipeline.py

# Quick development run (50K rows, faster)
python run_pipeline.py --quick
```

### 4. View results

All outputs are saved to `output/`:

| Path | Contents |
|------|----------|
| `output/figures/*.png` | 12 publication-quality charts |
| `output/data/feature_table.csv` | Master feature table |
| `output/data/signal_table.csv` | Trading signals |
| `output/data/trades.csv` | Individual trade records |
| `output/data/daily_pnl.csv` | Daily PnL series |
| `output/reports/performance_report.txt` | Full performance report |

---

## Pipeline Stages

The pipeline runs 8 stages sequentially:

### Stage 1 — Data Loading & Cleaning
- Loads OptionMetrics SPX weekly options (114K rows)
- Loads Treasury yield panel (16K days, 1-360 month maturities)
- Applies quality filters: min bid ($0.05), max spread (100%), DTE (7-60 days),
  moneyness band (±20%)
- Extracts daily SPX price series with log-returns

### Stage 2 — Feature Engineering
- **ATM IV (30d):** Interpolates near-the-money option IV to constant 30-day
  maturity using delta-weighted averaging
- **Model-Free IV:** Carr-Madan discrete variance swap approximation using OTM
  options within ±5% of forward
- **Realised Variance:** Rolling daily (1d), weekly (5d), and monthly (22d) RV
- **Bipower Variation:** Jump-robust integrated variance estimator
- **Forward RV:** Target variable (next 22-day realised variance)
- **FOMC Flags:** Binary event-window indicators

### Stage 3 — RV Forecasting Models
- **HAR-RV** (Corsi 2009): Heterogeneous autoregressive model using daily,
  weekly, monthly RV components
- **GARCH(1,1)** (Bollerslev 1986): Standard generalised autoregressive
  conditional heteroskedasticity
- **GJR-GARCH** (Glosten-Jagannathan-Runkle): Asymmetric leverage effect
- **Composite:** Equal-weight average of active models
- All models use expanding-window estimation (min 252 days)

### Stage 4 — Signal Construction
- **VRP Signal:** S = IV - forecast_RV, z-scored over rolling 252-day window
  - z > +1.0 → short vol (IV is rich)
  - z < -1.0 → long vol (IV is cheap)
- **Skew Signal:** Implied tail probability minus realised tail frequency
- **Term-Structure Signal:** Short-dated minus long-dated ATM IV
- **Distribution Signal:** Risk-neutral vs physical tail divergence

### Stage 5 — Backtesting
- Trades delta-hedged ATM straddles:
  - Short vol: sell straddle when VRP z-score > +1
  - Long vol: buy straddle when VRP z-score < -1
- Daily delta-hedging via Black-Scholes straddle delta
- 22-day default holding period
- Transaction costs: 5 bps per leg

### Stage 6 — Performance Analysis
- **Return metrics:** Sharpe, Sortino, Calmar ratios
- **Risk metrics:** Max drawdown, VaR, CVaR, skewness, kurtosis
- **Trade stats:** Win rate, profit factor, average PnL
- **Robustness:** Sub-period stability, parameter sensitivity grid
  (holding period × transaction cost)

### Stage 7 — Visualisation (12 figures)
1. SPX price level and daily returns
2. Options data summary (volume, IV distribution, moneyness, DTE)
3. IV vs realised volatility overlay with VRP spread
4. RV forecast models vs actual
5. VRP signal (level, z-score, discrete signal)
6. Surface signals (skew, term-structure, distribution)
7. Cumulative PnL and drawdown
8. Trade-level analysis (PnL distribution, direction breakdown, entry spread)
9. Monthly returns heatmap
10. Robustness by sub-period (Sharpe bars)
11. Parameter sensitivity heatmap (Sharpe across hold × cost)
12. Summary dashboard (6-panel overview)

### Stage 8 — Export
- All processed DataFrames saved to CSV
- Performance report saved as text file

---

## Key Results (Full Dataset)

| Metric | Value |
|--------|-------|
| Annualised Return | 15.5% |
| Annualised Volatility | 1.6% |
| Sharpe Ratio | 9.5 |
| Win Rate | 85.7% |
| Max Drawdown | -3.3% |
| Profit Factor | 5.55 |
| Number of Trades | 7 |

**Parameter sensitivity** shows the strategy is robust across holding periods
(15-30 days) and transaction cost assumptions (0-10 bps), with Sharpe ratios
ranging from 9.1 to 13.1.

---

## Module Documentation

### `src/config.py`
Central configuration. All paths, column mappings, and tunable parameters.
Modify this file to change data sources, filter thresholds, model parameters,
or trading rules.

### `src/data_loader.py`
- `load_options_raw()` — Read OptionMetrics CSV
- `clean_options()` — Apply quality filters, compute derived columns
- `extract_spx_prices()` — Daily SPX close from options data
- `load_yield_curve()` — Treasury zero-coupon yield panel
- `load_all_data()` — One-call convenience wrapper

### `src/feature_engineering.py`
- `compute_atm_iv()` — Delta-weighted ATM IV, constant-maturity interpolation
- `compute_model_free_iv()` — Carr-Madan variance swap
- `compute_realized_variance()` — Multi-horizon rolling RV
- `compute_bipower_variation()` — Jump-robust BV
- `compute_forward_rv()` — Forward-looking target RV
- `add_event_flags()` — FOMC/macro event windows
- `build_feature_table()` — Master feature table orchestrator

### `src/rv_models.py`
- `HARRV` class — fit/predict/summary for HAR-RV
- `fit_garch()` — Fit GARCH/EGARCH/GJR models via `arch` package
- `garch_rolling_forecast()` — Expanding-window GARCH forecasts
- `run_all_rv_models()` — Run all models, composite forecast

### `src/signals.py`
- `compute_vrp_signal()` — Core VRP z-score signal
- `compute_skew_signal()` — Skew mispricing (implied vs realised tail)
- `compute_term_structure_signal()` — Short vs long IV
- `compute_distribution_signal()` — RN vs physical distribution divergence
- `build_signal_table()` — All signals merged

### `src/backtest.py`
- `bs_straddle_price()` — Black-Scholes ATM straddle pricing
- `simulate_delta_hedge()` — Daily delta-hedge PnL simulation
- `run_backtest()` — Full entry/exit/hedge/cost backtest engine
- `trades_to_dataframe()` — Convert Trade objects to DataFrame

### `src/performance.py`
- `sharpe_ratio()`, `sortino_ratio()`, `calmar_ratio()`
- `compute_drawdown()` — Drawdown series and max DD
- `return_statistics()` — Full distribution stats (VaR, CVaR, skew, kurtosis)
- `trade_statistics()` — Trade-level metrics
- `full_performance_report()` — Comprehensive report
- `robustness_by_subperiod()` — Sub-sample stability
- `robustness_by_parameter()` — Hold period × cost sensitivity

### `src/visualization.py`
12 plotting functions generating publication-quality figures, all auto-saved
to `output/figures/`.

---

## References

- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized
  Volatility* (HAR-RV).
- Carr, P. & Madan, D. (1998). *Towards a Theory of Volatility Trading.*
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009). *Expected Stock Returns and
  Variance Risk Premia.*
- Carr, P. & Wu, L. (2009). *Variance Risk Premia.*
- Hansen, P.R., Huang, Z. & Shek, H.H. (2012). *Realized GARCH.*
- Breeden, D.T. & Litzenberger, R.H. (1978). *Prices of State-Contingent
  Claims Implicit in Option Prices.*
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.*

---

## License

Academic project — MS&E 244, Stanford University.
