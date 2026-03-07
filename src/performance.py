# ============================================================
# performance.py — Performance Metrics & Robustness Analysis
# ============================================================
"""
Computes standard strategy-evaluation metrics using empyrical
(Quantopian's backtesting library) for Sharpe, Sortino, drawdown,
Calmar, and related risk/return metrics.

Also includes:
    - Skewness / kurtosis of returns
    - Win rate, average trade PnL
    - Turnover / capacity estimates
    - Robustness across sub-periods and parameter variations

All functions accept the daily PnL or trade DataFrames produced
by the backtester.
"""

import numpy as np
import pandas as pd
import empyrical as ep

from src.config import TRADING_DAYS_PER_YEAR, NOTIONAL_CAPITAL


# ════════════════════════════════════════════════════════════
# 1.  RETURN-LEVEL METRICS (via empyrical)
# ════════════════════════════════════════════════════════════

def _to_series(x):
    """Ensure input is a pd.Series for empyrical."""
    if isinstance(x, np.ndarray):
        return pd.Series(x)
    return pd.Series(x) if not isinstance(x, pd.Series) else x


def annualised_return(daily_returns):
    """Annualised geometric return (CAGR) via empyrical."""
    r = _to_series(daily_returns).dropna()
    if len(r) < 2:
        return 0.0
    val = ep.annual_return(r, period="daily", annualization=TRADING_DAYS_PER_YEAR)
    return float(val) if not (isinstance(val, float) and np.isnan(val)) else 0.0


def annualised_volatility(daily_returns):
    """Annualised volatility via empyrical."""
    r = _to_series(daily_returns).dropna()
    if len(r) < 2:
        return 0.0
    val = ep.annual_volatility(r, period="daily", annualization=TRADING_DAYS_PER_YEAR)
    return float(val) if not (isinstance(val, float) and np.isnan(val)) else 0.0


def sharpe_ratio(daily_returns, risk_free_annual=0.0):
    """
    Annualised Sharpe ratio via empyrical.

    Parameters
    ----------
    daily_returns : pd.Series or np.ndarray
    risk_free_annual : float
        Annualised risk-free rate (converted to daily for empyrical).
    """
    r = _to_series(daily_returns).dropna()
    if len(r) < 2:
        return 0.0
    risk_free_daily = risk_free_annual / TRADING_DAYS_PER_YEAR
    val = ep.sharpe_ratio(r, risk_free=risk_free_daily, period="daily", annualization=TRADING_DAYS_PER_YEAR)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return float(val)


def sortino_ratio(daily_returns, risk_free_annual=0.0):
    """Sortino ratio via empyrical (penalises only downside volatility)."""
    r = _to_series(daily_returns).dropna()
    if len(r) < 2:
        return np.inf
    risk_free_daily = risk_free_annual / TRADING_DAYS_PER_YEAR
    val = ep.sortino_ratio(r, required_return=risk_free_daily, period="daily", annualization=TRADING_DAYS_PER_YEAR)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.inf
    return float(val)


# ════════════════════════════════════════════════════════════
# 2.  DRAWDOWN METRICS
# ════════════════════════════════════════════════════════════

def compute_drawdown(daily_returns):
    """
    Compute the drawdown series and max drawdown.
    Uses empyrical for max_drawdown; builds series for plotting.

    Parameters
    ----------
    daily_returns : pd.Series

    Returns
    -------
    dict with keys:
        drawdown_series : pd.Series (percentage drawdown from peak)
        max_drawdown    : float (worst drawdown, negative number)
        max_dd_start    : date/index when the peak before max DD occurred
        max_dd_end      : date/index when max drawdown was reached
    """
    r = _to_series(daily_returns).dropna()
    if r.empty:
        return {"drawdown_series": pd.Series(), "max_drawdown": 0.0, "max_dd_start": None, "max_dd_end": None}

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = np.where(peak > 1e-12, (cum - peak) / peak, 0.0)
    dd = pd.Series(dd, index=r.index)

    max_dd_val = ep.max_drawdown(r)
    max_dd = float(max_dd_val) if max_dd_val is not None and not np.isnan(max_dd_val) else dd.min()
    max_dd_end = dd.idxmin() if len(dd) > 0 else None
    max_dd_start = cum.loc[:max_dd_end].idxmax() if max_dd_end is not None else None

    return {
        "drawdown_series": dd,
        "max_drawdown": max_dd,
        "max_dd_start": max_dd_start,
        "max_dd_end": max_dd_end,
    }


def calmar_ratio(daily_returns):
    """Calmar ratio via empyrical (annual return / |max drawdown|)."""
    r = _to_series(daily_returns).dropna()
    if len(r) < 2:
        return np.inf
    val = ep.calmar_ratio(r, period="daily", annualization=TRADING_DAYS_PER_YEAR)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.inf
    return float(val)


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


def higher_moments_with_significance(daily_returns):
    """
    Higher moments (skewness, kurtosis) with standard errors and significance
    tests at the 5% level.

    Pandas ``.kurtosis()`` returns *excess* kurtosis (Fisher definition:
    0 for normal).  Fat tails = excess kurtosis *significantly* > 0, i.e.
    excess_kurtosis > 1.96 * SE(kurtosis).

    Returns
    -------
    dict with keys: skewness, kurtosis, excess_kurtosis,
                    skew_se, kurt_se, skew_significant, fat_tails
    """
    r = pd.Series(daily_returns).dropna()
    n = len(r)
    if n < 3:
        return {k: np.nan for k in [
            "skewness", "kurtosis", "excess_kurtosis",
            "skew_se", "kurt_se", "skew_significant", "fat_tails",
        ]}
    skew = r.skew()
    kurt = r.kurtosis()  # pandas: Fisher definition, 0 for normal
    excess_kurt = kurt  # already excess

    skew_se = np.sqrt(6.0 / n)
    kurt_se = np.sqrt(24.0 / n)

    skew_significant = bool(abs(skew) > 1.96 * skew_se)
    fat_tails = bool(excess_kurt > 0 and excess_kurt > 1.96 * kurt_se)

    return {
        "skewness": skew,
        "kurtosis": kurt,
        "excess_kurtosis": excess_kurt,
        "skew_se": skew_se,
        "kurt_se": kurt_se,
        "skew_significant": skew_significant,
        "fat_tails": fat_tails,
    }


def probabilistic_sharpe_ratio(daily_returns, sr_ref=0.0):
    """
    Probabilistic Sharpe Ratio (PSR): probability that true SR > sr_ref.

    Following Bailey & López de Prado (2012). PSR(SR*) = Prob(SR > SR*).
    Uses skewness and kurtosis to adjust the standard error of the Sharpe ratio.

    Parameters
    ----------
    daily_returns : pd.Series or np.ndarray
    sr_ref : float
        Reference Sharpe ratio (e.g. 0 for "probability that SR > 0").

    Returns
    -------
    float
        PSR in [0, 1].
    """
    from scipy import stats
    r = pd.Series(daily_returns).dropna()
    n = len(r)
    if n < 2:
        return np.nan

    mu = r.mean()
    sigma = r.std()
    if sigma == 0:
        return 1.0 if (mu > 0 and sr_ref <= 0) else 0.0

    # Annualised Sharpe (risk-free = 0)
    sr = (mu / sigma) * np.sqrt(TRADING_DAYS_PER_YEAR)

    skew = r.skew()
    kurt = r.kurtosis()  # excess kurtosis in pandas

    # Standard error of Sharpe under non-normality (López de Prado)
    term = 1.0 - skew * sr / np.sqrt(TRADING_DAYS_PER_YEAR) + (kurt - 1) / 4 * (sr ** 2) / TRADING_DAYS_PER_YEAR
    if term <= 0:
        term = 1e-10
    se_sr = np.sqrt(term / (n - 1))

    if se_sr <= 0:
        return 1.0 if sr > sr_ref else 0.0
    z = (sr - sr_ref) / se_sr
    return float(stats.norm.cdf(z))


# ════════════════════════════════════════════════════════════
# 3b. BENCHMARK (BUY-AND-HOLD SPX / SPY PROXY)
# ════════════════════════════════════════════════════════════

def benchmark_returns_from_spx(spx_df, start_date, end_date):
    """
    Get daily returns for buy-and-hold SPX over [start_date, end_date].

    SPX is used as proxy for SPY (they track closely). Aligns to strategy dates.

    Parameters
    ----------
    spx_df : pd.DataFrame
        Columns: date, spx_close.
    start_date, end_date : datetime-like

    Returns
    -------
    pd.Series
        Daily log return; index = date.
    """
    spx = spx_df[["date", "spx_close"]].copy()
    spx["date"] = pd.to_datetime(spx["date"]).dt.normalize()
    spx = spx.sort_values("date").drop_duplicates("date")
    spx = spx[(spx["date"] >= pd.Timestamp(start_date)) & (spx["date"] <= pd.Timestamp(end_date))]
    spx["return"] = spx["spx_close"].pct_change()  # simple return for comparability with strategy
    spx = spx.dropna(subset=["return"])
    return spx.set_index("date")["return"]


def benchmark_comparison(strategy_daily_returns, benchmark_daily_returns):
    """
    Compare strategy to buy-and-hold benchmark (e.g. SPX) using empyrical.

    Strategy returns are reindexed to the benchmark calendar: on days
    with no strategy trade, return = 0 (flat, capital parked). This
    ensures benchmark stats cover the full sample, not just strategy days.

    Returns
    -------
    dict with keys: benchmark_sharpe, benchmark_ann_return, benchmark_max_dd,
                   strategy_sharpe, strategy_ann_return, strategy_max_dd,
                   correlation, beta, alpha (annualised), information_ratio
    """
    b_ret = _to_series(benchmark_daily_returns).dropna()
    if b_ret.empty:
        return {}

    # Reindex strategy to benchmark calendar; fill missing days with 0 (flat)
    s_ret = _to_series(strategy_daily_returns).reindex(b_ret.index, fill_value=0.0).dropna()
    if s_ret.empty:
        return {}

    # Align for empyrical alpha_beta (same index)
    common = s_ret.index.intersection(b_ret.index)
    s_aligned = s_ret.loc[common].fillna(0.0)
    b_aligned = b_ret.loc[common].dropna()
    s_aligned = s_aligned.reindex(b_aligned.index).fillna(0.0)

    alpha_ann, beta = np.nan, np.nan
    if len(s_aligned) > 10 and len(b_aligned) > 10:
        try:
            alpha_ann, beta = ep.alpha_beta(
                s_aligned, b_aligned,
                risk_free=0.0, period="daily", annualization=TRADING_DAYS_PER_YEAR,
            )
            alpha_ann = float(alpha_ann) if alpha_ann is not None else np.nan
            beta = float(beta) if beta is not None else np.nan
        except Exception:
            pass

    ann_rf = 0.0
    sr_s = sharpe_ratio(s_ret, risk_free_annual=ann_rf)
    sr_b = sharpe_ratio(b_ret, risk_free_annual=ann_rf)
    ann_s = annualised_return(s_ret)
    ann_b = annualised_return(b_ret)
    if np.isnan(alpha_ann) and not np.isnan(beta):
        alpha_ann = ann_s - (ann_rf + beta * (ann_b - ann_rf))
    diff = s_ret - b_ret
    te = annualised_volatility(diff)
    ir = (ann_s - ann_b) / te if te and te > 0 else np.nan

    return {
        "benchmark_sharpe": sr_b,
        "benchmark_ann_return": ann_b,
        "benchmark_max_dd": compute_drawdown(b_ret)["max_drawdown"],
        "strategy_sharpe": sr_s,
        "strategy_ann_return": ann_s,
        "strategy_max_dd": compute_drawdown(s_ret)["max_drawdown"],
        "correlation": s_ret.corr(b_ret) if len(s_ret) > 1 else np.nan,
        "beta": beta,
        "alpha_ann": alpha_ann,
        "information_ratio": ir,
    }


# ════════════════════════════════════════════════════════════
# 4.  TRADE-LEVEL METRICS
# ════════════════════════════════════════════════════════════

def trade_statistics(trades_df):
    """
    Compute trade-level statistics from the trades DataFrame.

    Includes: win rate, PnL stats, holding period, trading frequency,
    gain per trade (avg, median, std), and capacity-related fields.

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
    holding = trades_df["holding_days"]

    n = len(trades_df)
    # Trading frequency: trades per year (use span of trades if available)
    trades_per_year = np.nan
    if "entry_date" in trades_df.columns and "exit_date" in trades_df.columns and n >= 1:
        try:
            entry_dates = pd.to_datetime(trades_df["entry_date"]).dropna()
            exit_dates = pd.to_datetime(trades_df["exit_date"]).dropna()
            if len(entry_dates) > 0 and len(exit_dates) > 0:
                span = exit_dates.max() - entry_dates.min()
                span_days = span.days if hasattr(span, "days") else float(span / np.timedelta64(1, "D"))
                span_years = max(span_days / 365.25, 1 / 365.25)
                trades_per_year = n / span_years
        except Exception:
            trades_per_year = np.nan

    # Guard against NaN in holding_days
    holding_clean = holding.dropna()
    avg_hold = holding_clean.mean() if len(holding_clean) > 0 else np.nan
    med_hold = holding_clean.median() if len(holding_clean) > 0 else np.nan
    min_hold = holding_clean.min() if len(holding_clean) > 0 else np.nan
    max_hold = holding_clean.max() if len(holding_clean) > 0 else np.nan

    iv_rv_mean = trades_df["iv_rv_spread"].mean() if "iv_rv_spread" in trades_df.columns else np.nan

    return {
        "n_trades": n,
        "win_rate": len(winners) / n,
        "avg_pnl": pnl.mean(),
        "median_pnl": pnl.median(),
        "std_pnl": pnl.std(),
        "total_pnl": pnl.sum(),
        "avg_winner": winners.mean() if len(winners) > 0 else 0,
        "avg_loser": losers.mean() if len(losers) > 0 else 0,
        "profit_factor": (
            winners.sum() / abs(losers.sum())
            if losers.sum() != 0 else np.inf
        ),
        "max_win": pnl.max(),
        "max_loss": pnl.min(),
        "avg_holding_days": avg_hold,
        "median_holding_days": med_hold,
        "min_holding_days": min_hold,
        "max_holding_days": max_hold,
        "trades_per_year": trades_per_year,
        "avg_iv_rv_spread": iv_rv_mean,
    }


def capacity_estimate(trades_df, notional_per_trade, adv_billions=50.0, pct_adv_max=0.01):
    """
    Illustrative capacity estimate: how much can we deploy without moving markets.

    This is an order-of-magnitude estimate, NOT data-driven. It assumes a
    fixed SPX options ADV and a maximum participation rate. For a real
    capacity study, use actual ADV from options volume data.

    Parameters
    ----------
    trades_df : pd.DataFrame
    notional_per_trade : float
        Typical notional per trade (e.g. from backtest config).
    adv_billions : float
        Assumed SPX options ADV in billions (order of magnitude, default 50).
    pct_adv_max : float
        Max fraction of ADV we allow (e.g. 0.01 = 1%).

    Returns
    -------
    dict with keys: capacity_usd, capacity_billions, scale_factor, note
    """
    capacity_usd = adv_billions * 1e9 * pct_adv_max
    capacity_billions = capacity_usd / 1e9
    scale_factor = capacity_usd / notional_per_trade if notional_per_trade and notional_per_trade > 0 else np.nan
    return {
        "capacity_usd": capacity_usd,
        "capacity_billions": capacity_billions,
        "scale_factor": scale_factor,
        "note": (f"Illustrative: assumes SPX options ADV ~${adv_billions:.0f}B; "
                 f"deploy up to {pct_adv_max:.1%} of ADV. Not data-driven."),
    }


# ════════════════════════════════════════════════════════════
# 5.  FULL PERFORMANCE REPORT
# ════════════════════════════════════════════════════════════

def returns_on_full_calendar(daily_pnl_df, spx_df=None):
    """
    Reindex strategy daily returns to the full trading-day calendar;
    fill days with no position with 0. This avoids inflating Sharpe by
    only counting "active" days when annualising.
    """
    strat_ret = daily_pnl_df.set_index("date")["daily_return"].dropna()
    if strat_ret.empty:
        return strat_ret
    start_date = strat_ret.index.min()
    end_date = strat_ret.index.max()
    if spx_df is not None and "date" in spx_df.columns:
        calendar = spx_df.loc[
            (spx_df["date"] >= start_date) & (spx_df["date"] <= end_date),
            "date",
        ].drop_duplicates()
        full_dates = pd.DatetimeIndex(calendar.sort_values().values)
    else:
        full_dates = pd.bdate_range(start_date, end_date)
    return strat_ret.reindex(full_dates, fill_value=0.0)


def full_performance_report(daily_pnl_df, trades_df, spx_df=None, notional=NOTIONAL_CAPITAL):
    """
    Generate a comprehensive performance report.

    Parameters
    ----------
    daily_pnl_df : pd.DataFrame
        From run_backtest(), columns: date, daily_return, …
    trades_df : pd.DataFrame
        From trades_to_dataframe().
    spx_df : pd.DataFrame, optional
        SPX price series (date, spx_close) for buy-and-hold benchmark.
    notional : float
        Notional per trade for capacity estimate.

    Returns
    -------
    dict
        Nested dictionary of all metrics. If daily_return is empty or has
        fewer than 2 observations, return metrics and drawdown use NaN/0
        and a short notice is printed.
    """
    dr = daily_pnl_df["daily_return"].dropna()
    if dr.empty or len(dr) < 2:
        report = {
            "return_metrics": {
                "annualised_return": np.nan,
                "annualised_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "sortino_ratio": np.nan,
                "calmar_ratio": np.nan,
            },
            "drawdown": {"drawdown_series": dr, "max_drawdown": 0.0, "max_dd_start": None, "max_dd_end": None},
            "distribution": return_statistics(dr) if len(dr) > 0 else {},
            "trade_stats": trade_statistics(trades_df),
            "higher_moments": higher_moments_with_significance(dr),
            "probabilistic_sharpe_ratio": np.nan,
            "benchmark": {},
            "capacity": capacity_estimate(trades_df, notional),
        }
        print("\n  [performance] Insufficient daily returns (need ≥2); report uses defaults for return metrics.")
        return report

    # Use full calendar (flat days = 0) so ann. return and Sharpe are not inflated
    dr_calendar = returns_on_full_calendar(daily_pnl_df, spx_df)
    if dr_calendar.empty or len(dr_calendar) < 2:
        dr_calendar = dr
    else:
        dr_calendar = dr_calendar.dropna()

    report = {
        "return_metrics": {
            "annualised_return": annualised_return(dr_calendar),
            "annualised_volatility": annualised_volatility(dr_calendar),
            "sharpe_ratio": sharpe_ratio(dr_calendar),
            "sortino_ratio": sortino_ratio(dr_calendar),
            "calmar_ratio": calmar_ratio(dr_calendar),
        },
        "drawdown": compute_drawdown(dr_calendar),
        "distribution": return_statistics(dr_calendar),
        "trade_stats": trade_statistics(trades_df),
        "higher_moments": higher_moments_with_significance(dr_calendar),
        "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio(dr_calendar, sr_ref=0.0),
        "benchmark": {},
        "capacity": capacity_estimate(trades_df, notional),
    }

    # Strategy returns for benchmark alignment (use calendar series so benchmark sees same span)
    if "date" in daily_pnl_df.columns:
        strat_ret = daily_pnl_df.set_index("date")["daily_return"].dropna()
    else:
        strat_ret = dr

    if spx_df is not None and len(strat_ret) > 0:
        start_date = strat_ret.index.min()
        end_date = strat_ret.index.max()
        bench_ret = benchmark_returns_from_spx(spx_df, start_date, end_date)
        if len(bench_ret) > 0:
            # Align strategy to benchmark calendar (flat days = 0)
            s_ret = dr_calendar.reindex(bench_ret.index, fill_value=0.0).dropna()
            if len(s_ret) > 0:
                report["benchmark"] = benchmark_comparison(s_ret, bench_ret)
            else:
                report["benchmark"] = benchmark_comparison(strat_ret, bench_ret)

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

    # Probabilistic Sharpe Ratio
    psr = report["probabilistic_sharpe_ratio"]
    print(f"  Prob. Sharpe (SR>0):   {psr:>10.2%}")

    ts = report["trade_stats"]
    if ts["n_trades"] > 0:
        print(f"\n  --- Trades ---")
        print(f"  Trades:        {ts['n_trades']}")
        print(f"  Trades/year:   {ts.get('trades_per_year', np.nan):.1f}")
        print(f"  Win Rate:      {ts['win_rate']:.1%}")
        print(f"  Avg PnL:       ${ts['avg_pnl']:,.2f}  (gain per trade)")
        print(f"  Median PnL:    ${ts['median_pnl']:,.2f}")
        print(f"  Total PnL:     ${ts['total_pnl']:,.2f}")
        print(f"  Profit Factor: {ts['profit_factor']:.2f}")
        print(f"  Avg holding:   {ts.get('avg_holding_days', np.nan):.1f} days  "
              f"(min={ts.get('min_holding_days', np.nan):.0f}, max={ts.get('max_holding_days', np.nan):.0f})")

    # Higher moments
    hm = report["higher_moments"]
    print(f"\n  --- Higher moments ---")
    print(f"  Skewness:      {hm['skewness']:.3f}  (SE={hm['skew_se']:.3f})  "
          f"Significant: {hm['skew_significant']}")
    print(f"  Kurtosis:      {hm['kurtosis']:.3f}  (excess={hm['excess_kurtosis']:.3f})  "
          f"Fat tails:   {hm['fat_tails']}")

    ds = report["distribution"]
    print(f"  VaR (95%):    {ds['var_95']:.4f}")
    print(f"  CVaR(95%):    {ds['cvar_95']:.4f}")

    # Benchmark
    if report["benchmark"]:
        b = report["benchmark"]
        print(f"\n  --- Benchmark (buy-and-hold SPX) ---")
        print(f"  Benchmark Sharpe:   {b['benchmark_sharpe']:.2f}")
        print(f"  Benchmark Ann Ret: {b['benchmark_ann_return']:.2%}")
        print(f"  Benchmark Max DD:  {b['benchmark_max_dd']:.2%}")
        print(f"  Correlation:       {b['correlation']:.3f}")
        print(f"  Beta:              {b['beta']:.3f}")
        print(f"  Alpha (ann):       {b['alpha_ann']:.2%}")
        print(f"  Info Ratio:        {b['information_ratio']:.3f}")

    # Capacity
    cap = report["capacity"]
    print(f"\n  --- Capacity ---")
    print(f"  {cap['note']}")
    print(f"  Capacity:      ${cap['capacity_usd']/1e6:.1f}M ({cap['capacity_billions']:.2f}B)")
    print(f"  Scale factor: {cap['scale_factor']:.0f}x vs ${notional/1e6:.1f}M/trade")

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
    For weeklies, hold_days_range should be <= 5 (backtest never holds past expiry).

    Parameters
    ----------
    signal_df, spx_df : pd.DataFrame
    backtest_fn : callable
        The run_backtest function.
    hold_days_range : list of int
        Holding periods to test. For Friday-expiring weeklies use <= 5 (never hold post-expiry).
    cost_range : list of float (bps)

    Returns
    -------
    pd.DataFrame
        Parameter grid with performance metrics.
    """
    hold_days_range = hold_days_range or [1, 2, 3, 4, 5]  # weeklies: max 5 trading days to expiry
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
