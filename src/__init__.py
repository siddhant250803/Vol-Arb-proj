# ============================================================
# src/ — IV vs Forecast RV Volatility Arbitrage Pipeline
# MS&E 244, Stanford University
# ============================================================
"""
Package layout
--------------
config.py               Global paths, constants, and parameters
data_loader.py          Load & clean SPX options, yields, prices
feature_engineering.py  ATM IV, model-free IV, realized volatility
rv_models.py            HAR-RV, GARCH/EGARCH/GJR-GARCH forecasts
signals.py              VRP signal, skew & term-structure signals
backtest.py             Delta-hedged straddle backtesting engine
performance.py          Sharpe, Sortino, drawdowns, robustness
visualization.py        All plotting utilities & dashboards
"""
