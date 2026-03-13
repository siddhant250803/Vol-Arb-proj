"""
IV vs Forecast RV — Volatility Arbitrage Pipeline

Modules
-------
config                 Global paths, constants, and parameters
data_loader            Load & clean SPX options, yields, prices
feature_engineering    ATM IV, model-free IV, realized volatility
rv_models              HAR-RV, GARCH, GJR-GARCH forecasts
signals                VRP signal, skew & term-structure signals
backtest               Delta-hedged straddle backtesting engine
performance            Sharpe, Sortino, drawdowns, robustness
visualization          All plotting utilities & dashboards
"""
