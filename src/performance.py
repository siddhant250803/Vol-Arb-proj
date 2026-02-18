# ============================================================
# performance.py — Performance Metrics & Robustness Analysis
# ============================================================
"""
Computes standard strategy-evaluation metrics:

    - Sharpe ratio, Sortino ratio
    - Maximum drawdown, Calmar ratio
    - Skewness / kurtosis of returns
    - Win rate, average trade PnL
    - Turnover / capacity estimates
    - Robustness across sub-periods and parameter variations

All functions accept the daily PnL or trade DataFrames produced
by the backtester.
"""

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


# ════════════════════════════════════════════════════════════
# 1.  RETURN-LEVEL METRICS
# ════════════════════════════════════════════════════════════

def annualised_return(daily_returns):
    """Annualised geometric return from a series of daily returns."""
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return total ** (1.0 / n_years) - 1.0


def annualised_volatility(daily_returns):
    """Annualised volatility (standard deviation) of daily returns."""
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns, risk_free_annual=0.0):
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    daily_returns : pd.Series or np.ndarray
    risk_free_annual : float
        Annualised risk-free rate.
    """
    ann_ret = annualised_return(daily_returns)
    ann_vol = annualised_volatility(daily_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_annual) / ann_vol


def sortino_ratio(daily_returns, risk_free_annual=0.0):
    """
    Sortino ratio: penalises only downside volatility.

    Downside deviation = std of returns below zero (annualised).
    """
    ann_ret = annualised_return(daily_returns)
    downside = daily_returns[daily_returns < 0]
    if len(downside) == 0:
        return np.inf
    dd = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if dd == 0:
        return np.inf
    return (ann_ret - risk_free_annual) / dd


# ════════════════════════════════════════════════════════════
# 2.  DRAWDOWN METRICS
# ════════════════════════════════════════════════════════════

def compute_drawdown(daily_returns):
    """
    Compute the drawdown series and max drawdown.

    Parameters
    ----------
    daily_returns : pd.Series

    Returns
    -------
    dict with keys:
        drawdown_series : pd.Series (percentage drawdown from peak)
        max_drawdown    : float (worst drawdown, negative number)
        max_dd_start    : date when drawdown period began
        max_dd_end      : date when max drawdown was reached
    """
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    max_dd = dd.min()
    max_dd_end = dd.idxmin()

    # Find the start of the drawdown period
    peak_before = cum.loc[:max_dd_end].idxmax() if hasattr(dd.index, 'get_loc') else 0

    return {
        "drawdown_series": dd,
        "max_drawdown": max_dd,
        "max_dd_end": max_dd_end,
    }


def calmar_ratio(daily_returns):
    """Calmar ratio = annualised return / |max drawdown|."""
    ann_ret = annualised_return(daily_returns)
    dd_info = compute_drawdown(daily_returns)
    max_dd = abs(dd_info["max_drawdown"])
    if max_dd == 0:
        return np.inf
    return ann_ret / max_dd


# ════════════════════════════════════════════════════════════
# 3.  DISTRIBUTION METRICS
# ════════════════════════════════════════════════════════════

def return_statistics(daily_returns):
    """
    Comprehensive return distribution statistics.

    Returns
    -------
    dict with keys: mean, std, skew, kurtosis, min, max,
                    pct_positive, var_95, cvar_95
    """
    r = pd.Series(daily_returns)
    var_95 = r.quantile(0.05)       # 5th percentile (VaR)
    cvar_95 = r[r <= var_95].mean() if (r <= var_95).any() else var_95

    return {
        "mean_daily": r.mean(),
        "std_daily": r.std(),
        "skewness": r.skew(),
        "kurtosis": r.kurtosis(),
        "min_daily": r.min(),
        "max_daily": r.max(),
        "pct_positive": (r > 0).mean(),
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


# ════════════════════════════════════════════════════════════
# 4.  TRADE-LEVEL METRICS
# ════════════════════════════════════════════════════════════

def trade_statistics(trades_df):
    """
    Compute trade-level statistics from the trades DataFrame.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Output of ``trades_to_dataframe()``.

    Returns
    -------
    dict
    """
    if trades_df.empty:
        return {"n_trades": 0}

    pnl = trades_df["net_pnl"]
    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]

    return {
        "n_trades": len(trades_df),
        "win_rate": len(winners) / len(trades_df),
        "avg_pnl": pnl.mean(),
        "median_pnl": pnl.median(),
        "total_pnl": pnl.sum(),
        "avg_winner": winners.mean() if len(winners) > 0 else 0,
        "avg_loser": losers.mean() if len(losers) > 0 else 0,
        "profit_factor": (
            winners.sum() / abs(losers.sum())
            if losers.sum() != 0 else np.inf
        ),
        "max_win": pnl.max(),
        "max_loss": pnl.min(),
        "avg_holding_days": trades_df["holding_days"].mean(),
        "avg_iv_rv_spread": trades_df["iv_rv_spread"].mean(),
    }


# ════════════════════════════════════════════════════════════
# 5.  FULL PERFORMANCE REPORT
# ════════════════════════════════════════════════════════════

def full_performance_report(daily_pnl_df, trades_df):
    """
    Generate a comprehensive performance report.

    Parameters
    ----------
    daily_pnl_df : pd.DataFrame
        From run_backtest(), columns: date, daily_return, …
    trades_df : pd.DataFrame
        From trades_to_dataframe().

    Returns
    -------
    dict
        Nested dictionary of all metrics.
    """
    dr = daily_pnl_df["daily_return"].dropna()

    report = {
        "return_metrics": {
            "annualised_return": annualised_return(dr),
            "annualised_volatility": annualised_volatility(dr),
            "sharpe_ratio": sharpe_ratio(dr),
            "sortino_ratio": sortino_ratio(dr),
            "calmar_ratio": calmar_ratio(dr),
        },
        "drawdown": compute_drawdown(dr),
        "distribution": return_statistics(dr),
        "trade_stats": trade_statistics(trades_df),
    }

    # Pretty-print summary
    print("\n" + "=" * 60)
    print("  PERFORMANCE REPORT")
    print("=" * 60)

    rm = report["return_metrics"]
    print(f"  Annualised Return:     {rm['annualised_return']:>10.2%}")
    print(f"  Annualised Volatility: {rm['annualised_volatility']:>10.2%}")
    print(f"  Sharpe Ratio:          {rm['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:         {rm['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:          {rm['calmar_ratio']:>10.2f}")
    print(f"  Max Drawdown:          {report['drawdown']['max_drawdown']:>10.2%}")

    ts = report["trade_stats"]
    if ts["n_trades"] > 0:
        print(f"\n  Trades:   {ts['n_trades']}")
        print(f"  Win Rate: {ts['win_rate']:.1%}")
        print(f"  Avg PnL:  ${ts['avg_pnl']:,.2f}")
        print(f"  Total PnL: ${ts['total_pnl']:,.2f}")
        print(f"  Profit Factor: {ts['profit_factor']:.2f}")

    ds = report["distribution"]
    print(f"\n  Skewness:  {ds['skewness']:.3f}")
    print(f"  Kurtosis:  {ds['kurtosis']:.3f}")
    print(f"  VaR (95%): {ds['var_95']:.4f}")
    print(f"  CVaR(95%): {ds['cvar_95']:.4f}")
    print("=" * 60)

    return report


# ════════════════════════════════════════════════════════════
# 6.  ROBUSTNESS TESTS
# ════════════════════════════════════════════════════════════

def robustness_by_subperiod(daily_pnl_df, n_periods=4):
    """
    Split the sample into n sub-periods and compute metrics for each.

    Parameters
    ----------
    daily_pnl_df : pd.DataFrame
    n_periods : int

    Returns
    -------
    pd.DataFrame
        One row per sub-period with Sharpe, return, max DD.
    """
    dr = daily_pnl_df[["date", "daily_return"]].dropna()
    chunk_size = len(dr) // n_periods
    results = []

    for i in range(n_periods):
        start = i * chunk_size
        end = start + chunk_size if i < n_periods - 1 else len(dr)
        chunk = dr.iloc[start:end]
        r = chunk["daily_return"]
        results.append({
            "period": i + 1,
            "start": chunk["date"].iloc[0],
            "end": chunk["date"].iloc[-1],
            "ann_return": annualised_return(r),
            "ann_vol": annualised_volatility(r),
            "sharpe": sharpe_ratio(r),
            "max_dd": compute_drawdown(r)["max_drawdown"],
            "n_days": len(r),
        })

    return pd.DataFrame(results)


def robustness_by_parameter(signal_df, spx_df, backtest_fn,
                            hold_days_range=None, cost_range=None):
    """
    Vary holding period and cost assumptions; report Sharpe for each.

    Parameters
    ----------
    signal_df, spx_df : pd.DataFrame
    backtest_fn : callable
        The run_backtest function.
    hold_days_range : list of int
    cost_range : list of float (bps)

    Returns
    -------
    pd.DataFrame
        Parameter grid with performance metrics.
    """
    hold_days_range = hold_days_range or [10, 15, 22, 30, 44]
    cost_range = cost_range or [0, 3, 5, 10, 15]

    results = []
    for hd in hold_days_range:
        for cost in cost_range:
            trades, pnl_df = backtest_fn(signal_df, spx_df,
                                         hold_days=hd, cost_bps=cost)
            if pnl_df.empty or len(pnl_df) < 10:
                continue
            dr = pnl_df["daily_return"]
            results.append({
                "hold_days": hd,
                "cost_bps": cost,
                "sharpe": sharpe_ratio(dr),
                "ann_return": annualised_return(dr),
                "max_dd": compute_drawdown(dr)["max_drawdown"],
                "n_trades": len(trades),
            })

    return pd.DataFrame(results)
