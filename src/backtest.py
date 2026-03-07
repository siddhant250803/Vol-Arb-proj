# ============================================================
# backtest.py — Delta-Hedged Straddle Backtesting Engine
# ============================================================
"""
Simulates a volatility-trading strategy that:

    1. Enters delta-hedged ATM straddles when signal is extreme
    2. Delta-hedges at a configurable frequency
    3. Exits at expiry or at a fixed horizon (never holds past option expiry)
    4. Accounts for bid-ask spreads and transaction costs

When signals carry exdate_trade, positions are closed at or before that expiry.
Otherwise hold is capped at days to next Friday (weeklies). No position is ever
held past the option expiry date.

The core abstraction is a ``Trade`` dataclass that tracks each
position from entry to exit, computing component PnLs.

**Daily PnL note:** each trade's net PnL is spread *evenly* across its
holding days (amortized attribution). This is a simplification — it does
not reflect intra-trade mark-to-market. Sharpe, drawdown, and other
return-based stats computed from this series are therefore approximations.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import (
    NOTIONAL_CAPITAL,
    CONTRACT_MULTIPLIER,
    TRANSACTION_COST_BPS,
    POSITION_HOLD_DAYS,
    TRADING_DAYS_PER_YEAR,
    SIGNAL_ZSCORE_ENTRY,
    STOP_LOSS_PCT,
)


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Represents one completed straddle trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int              # +1 = long vol, −1 = short vol
    n_contracts: int            # number of straddle contracts traded
    entry_iv: float
    realised_vol: float         # actual RV over holding period
    entry_price: float          # SPX level at entry
    exit_price: float           # SPX level at exit
    straddle_premium: float     # premium per contract (×multiplier)
    option_pnl: float           # PnL from option payoff vs premium
    hedge_pnl: float            # PnL from delta-hedging
    txn_cost: float             # total transaction costs
    net_pnl: float              # option_pnl + hedge_pnl − txn_cost
    holding_days: int
    stopped_out: bool = False   # True if exited via stop-loss


# ────────────────────────────────────────────────────────────
# 1.  Approximate Straddle Pricing (Black-Scholes)
# ────────────────────────────────────────────────────────────

def bs_straddle_price(S, K, T, sigma, r=0.0):
    """
    Black-Scholes ATM straddle price (call + put).

    For ATM (K ≈ S), straddle ≈ 2 · S · σ · √(T/(2π)) approximately.

    Parameters
    ----------
    S : float  — Spot price
    K : float  — Strike price
    T : float  — Time to expiry in years
    sigma : float — Implied volatility
    r : float  — Risk-free rate

    Returns
    -------
    float — Straddle price (call premium + put premium)
    """
    from scipy.stats import norm as sp_norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-10)
    d2 = d1 - sigma * np.sqrt(T)

    call = S * sp_norm.cdf(d1) - K * np.exp(-r * T) * sp_norm.cdf(d2)
    put = K * np.exp(-r * T) * sp_norm.cdf(-d2) - S * sp_norm.cdf(-d1)

    return call + put


# ────────────────────────────────────────────────────────────
# 2.  Delta-Hedging Simulation
# ────────────────────────────────────────────────────────────

def simulate_delta_hedge(prices, sigma, T_years, direction):
    """
    Simulate daily delta-hedging of an ATM straddle.

    The delta of a straddle = delta_call + delta_put.
    For ATM, net straddle delta ≈ 0 at inception, but drifts.

    We track the PnL from adjusting the hedge position daily.

    Parameters
    ----------
    prices : np.ndarray
        Daily SPX prices over the holding period.
    sigma : float
        Implied volatility at entry.
    T_years : float
        Time to expiry at entry (in years).
    direction : int
        +1 if long the straddle, −1 if short.

    Returns
    -------
    float
        Net hedge PnL.
    """
    from scipy.stats import norm as sp_norm

    n = len(prices)
    if n < 2:
        return 0.0

    K = prices[0]  # ATM strike
    hedge_pnl = 0.0
    prev_delta = 0.0

    for i in range(n - 1):
        S = prices[i]
        tau = T_years - (i / TRADING_DAYS_PER_YEAR)
        if tau <= 0:
            break

        # Straddle delta = delta_call + delta_put = 2·N(d1) − 1
        d1 = (np.log(S / K) + (0.5 * sigma ** 2) * tau) / (
            sigma * np.sqrt(tau) + 1e-10
        )
        straddle_delta = 2 * sp_norm.cdf(d1) - 1.0

        # Net delta exposure (if long straddle, we are long delta)
        target_delta = direction * straddle_delta
        # Hedge: go short target_delta shares of underlying
        hedge_pos = -target_delta

        # PnL from previous hedge position
        price_change = prices[i + 1] - prices[i]
        hedge_pnl += prev_delta * price_change

        prev_delta = hedge_pos

    return hedge_pnl


# ────────────────────────────────────────────────────────────
# 2b. Days to Next Friday (for Friday-expiring weeklies)
# ────────────────────────────────────────────────────────────

def trading_days_until_next_friday(d: pd.Timestamp) -> int:
    """
    Trading days from date d until the next Friday (inclusive of Friday).
    Monday → Friday = 4 days, Friday → next Friday = 5 days.
    """
    w = d.weekday()  # 0=Mon, 4=Fri
    if w == 4:  # Friday: next expiry is next week
        return 5
    return 4 - w  # Mon:4, Tue:3, Wed:2, Thu:1


def effective_hold_days(entry_date: pd.Timestamp, requested_hold: int) -> int:
    """
    Cap holding period at days until next Friday (options expire every Friday).
    Fallback when signal has no exdate_trade.
    """
    days_to_friday = trading_days_until_next_friday(entry_date)
    return min(requested_hold, days_to_friday)


def hold_days_from_expiry(entry_date: pd.Timestamp, exdate) -> Optional[int]:
    """
    Trading days from entry_date to exdate (inclusive).
    Returns None if exdate is missing or already passed.
    """
    if exdate is None:
        return None
    try:
        ex = pd.Timestamp(exdate)
    except Exception:
        return None
    if pd.isna(ex) or entry_date > ex:
        return None  # missing or already past expiry
    return len(pd.bdate_range(entry_date, ex))


# ────────────────────────────────────────────────────────────
# 3.  Core Backtester
# ────────────────────────────────────────────────────────────

def run_backtest(signal_df, spx_df, hold_days=None, cost_bps=None,
                 notional=None, stop_loss_pct=None):
    """
    Run the full backtest: enter on signal, hold, exit, compute PnL.

    PnL marking convention:
        - Hold to expiry: option PnL is marked to the closing price at the end
          of the week when the option expires (strike = entry price).
        - Early exit (stop-loss / mid-week): option PnL is marked to the closing
          price on the exit day; straddle is valued at (spot = exit close,
          strike = entry strike, time remaining).

    Position sizing:
        For each trade, the number of straddle contracts is computed as:
            n_contracts = floor(notional / (straddle_price × multiplier))
        All PnLs are scaled by n_contracts × multiplier.

    Stop-loss: exit when unrealized PnL (mark-to-market option + hedge) falls
    below -stop_loss_pct × trade value (e.g. 25% → exit when loss ≥ 25% of
    that trade's premium/deployed value, not 25% of portfolio notional).

    Strategy logic:
        - signal = +1  →  SHORT vol (sell straddle)  — expect IV to compress
        - signal = −1  →  LONG vol  (buy straddle)   — expect vol spike
        - signal =  0  →  no trade

    PnL decomposition:
        option_pnl  = n × multiplier × (straddle_at_entry − straddle_payoff or MTM)
        hedge_pnl   = n × multiplier × (PnL from daily delta rebalancing)
        net_pnl     = option_pnl + hedge_pnl − transaction_costs

    Parameters
    ----------
    signal_df : pd.DataFrame
        Must have columns: date, signal, iv, rv_forecast.
    spx_df : pd.DataFrame
        Daily SPX prices with columns: date, spx_close.
    hold_days : int
        Holding period in trading days.
    cost_bps : float
        One-way transaction cost in basis points.
    notional : float
        Total capital to deploy per trade (default: NOTIONAL_CAPITAL).
    stop_loss_pct : float, optional
        Exit when unrealized loss reaches this fraction of the trade's value
        (premium deployed on that position). E.g. 0.25 = 25% of trade value.
        Default from config STOP_LOSS_PCT.

    Returns
    -------
    tuple of (trades_list, daily_pnl_df)
        trades_list : List[Trade] — completed trades
        daily_pnl_df : pd.DataFrame — daily strategy returns
    """
    hold_days = hold_days or POSITION_HOLD_DAYS
    cost_bps = cost_bps or TRANSACTION_COST_BPS
    notional = notional or NOTIONAL_CAPITAL
    stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else STOP_LOSS_PCT
    multiplier = CONTRACT_MULTIPLIER

    # Merge SPX prices with signals
    merged = signal_df.merge(
        spx_df[["date", "spx_close"]],
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    n = len(merged)
    trades: List[Trade] = []
    daily_pnl = []

    # Track open position
    in_position = False
    entry_idx = 0

    i = 0
    while i < n:
        row = merged.iloc[i]

        if not in_position and row["signal"] != 0:
            # ── ENTRY ──────────────────────────────────────
            entry_date = row["date"]
            exdate = row.get("exdate_trade", None)

            # Cannot trade past expiry
            if exdate is not None and not pd.isna(exdate):
                if entry_date >= pd.Timestamp(exdate):
                    i += 1
                    continue  # skip: signal is post-expiry

            # Hold period from expiry if signal carries it, else cap at Friday
            this_hold = hold_days_from_expiry(entry_date, exdate)
            if this_hold is None or this_hold <= 0:
                this_hold = effective_hold_days(entry_date, hold_days)
            entry_exdate = pd.Timestamp(exdate) if exdate is not None and not pd.isna(exdate) else None

            in_position = True
            entry_idx = i
            direction = int(row["signal"])
            entry_iv = row["iv"]
            entry_price = row["spx_close"]

            # Straddle premium: use actual time to expiry when known
            if entry_exdate is not None:
                entry_T = (entry_exdate - entry_date).days / 365.0
            else:
                entry_T = this_hold / TRADING_DAYS_PER_YEAR
            straddle_per_share = bs_straddle_price(
                entry_price, entry_price, entry_T, entry_iv
            )

            # Position sizing: how many contracts can we trade?
            cost_per_contract = straddle_per_share * multiplier
            n_contracts = max(1, int(notional / cost_per_contract))

            i += 1
            continue

        if in_position:
            days_held = i - entry_idx
            price_path = merged.iloc[entry_idx: i + 1]["spx_close"].values
            scale = n_contracts * multiplier
            # Time to expiry at entry (for hedge/MTM)
            T_full = entry_T

            # ── Stop-loss: exit when loss reaches 25% of this trade's value ─
            # Trade value = premium deployed (straddle_per_share × scale), not portfolio notional
            # MTM must use *current* vol (realized vol so far), not entry_iv, so when vol spikes
            # we see the true loss and the stop triggers (otherwise e.g. Volmageddon never stops).
            trade_value = straddle_per_share * scale
            # Exit at expiry, horizon, or end of data
            at_expiry = (
                entry_exdate is not None
                and row["date"] >= entry_exdate
            )
            exit_now = (
                at_expiry
                or days_held >= this_hold       # fixed horizon
                or i == n - 1                    # end of data
            )
            if not exit_now and days_held >= 1 and stop_loss_pct is not None and stop_loss_pct > 0:
                current_spot = price_path[-1]
                tau_remaining = T_full - (len(price_path) - 1) / TRADING_DAYS_PER_YEAR
                if tau_remaining > 1e-6 and len(price_path) >= 2:
                    log_rets = np.diff(np.log(price_path))
                    rv_so_far = np.sqrt(
                        np.sum(log_rets ** 2) / len(log_rets) * TRADING_DAYS_PER_YEAR
                    )
                    mtm_sigma = max(float(entry_iv), float(rv_so_far))
                    mtm_straddle = bs_straddle_price(
                        current_spot, entry_price, tau_remaining, mtm_sigma
                    )
                    if direction == -1:
                        opt_unrealized_per = straddle_per_share - mtm_straddle
                    else:
                        opt_unrealized_per = mtm_straddle - straddle_per_share
                    hedge_so_far = simulate_delta_hedge(
                        price_path, entry_iv, T_full, direction
                    )
                    running_pnl = (opt_unrealized_per + hedge_so_far) * scale
                    if running_pnl <= -stop_loss_pct * trade_value:
                        exit_now = True

            if exit_now:
                # Always mark PnL to closing price: expiry = end-of-week close, early = close on exit day
                exit_price = row["spx_close"]  # closing price on exit date
                exit_date = row["date"]

                # Early exit (mid-week): mark to closing price at point of exit; strike = entry strike
                # Normal exit (expiry): mark to closing price at end of week when option expires
                is_early_exit = (entry_exdate is not None and not at_expiry) or (
                    entry_exdate is None and days_held < this_hold and i < n - 1
                )
                if is_early_exit:
                    # Mid-week exit: value straddle at (spot = closing price at exit, strike = entry_price)
                    tau_remaining = T_full - days_held / TRADING_DAYS_PER_YEAR
                    tau_remaining = max(tau_remaining, 1e-6)
                    if len(price_path) >= 2:
                        log_rets = np.diff(np.log(price_path))
                        rv_at_exit = np.sqrt(
                            np.sum(log_rets ** 2) / len(log_rets) * TRADING_DAYS_PER_YEAR
                        )
                        mtm_sigma = max(float(entry_iv), float(rv_at_exit))
                    else:
                        mtm_sigma = entry_iv
                    mtm_straddle = bs_straddle_price(
                        exit_price, entry_price, tau_remaining, mtm_sigma
                    )
                    if direction == -1:
                        opt_pnl_per = straddle_per_share - mtm_straddle
                    else:
                        opt_pnl_per = mtm_straddle - straddle_per_share
                    option_pnl = opt_pnl_per * scale
                    hedge_pnl_per = simulate_delta_hedge(
                        price_path, entry_iv, T_full, direction
                    )
                    hedge_pnl = hedge_pnl_per * scale
                else:
                    # Expiry: mark to closing price at end of week when option expires (strike = entry)
                    call_payoff = max(exit_price - entry_price, 0)
                    put_payoff = max(entry_price - exit_price, 0)
                    straddle_payoff = call_payoff + put_payoff
                    if direction == -1:
                        opt_pnl_per = straddle_per_share - straddle_payoff
                    else:
                        opt_pnl_per = straddle_payoff - straddle_per_share
                    option_pnl = opt_pnl_per * scale
                    hedge_pnl_per = simulate_delta_hedge(
                        price_path, entry_iv, T_full, direction
                    )
                    hedge_pnl = hedge_pnl_per * scale

                # Transaction costs (on full notional deployed)
                notional_deployed = straddle_per_share * scale
                txn = (cost_bps / 10000.0) * notional_deployed * 2  # entry + exit

                # Realised vol over holding period
                if len(price_path) > 1:
                    log_rets = np.diff(np.log(price_path))
                    rv = np.sqrt(
                        np.sum(log_rets ** 2)
                        * TRADING_DAYS_PER_YEAR / len(log_rets)
                    )
                else:
                    rv = 0.0

                net = option_pnl + hedge_pnl - txn
                # Cap realized loss on stop-loss exits at stop_loss_pct of trade value
                if is_early_exit and stop_loss_pct > 0 and net < -stop_loss_pct * notional_deployed:
                    net = -stop_loss_pct * notional_deployed

                trade = Trade(
                    entry_date=entry_date,
                    exit_date=exit_date,
                    direction=direction,
                    n_contracts=n_contracts,
                    entry_iv=entry_iv,
                    realised_vol=rv,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    straddle_premium=straddle_per_share * multiplier,
                    option_pnl=option_pnl,
                    hedge_pnl=hedge_pnl,
                    txn_cost=txn,
                    net_pnl=net,
                    holding_days=days_held,
                    stopped_out=is_early_exit,
                )
                trades.append(trade)

                # Amortized daily PnL: net PnL spread evenly over calendar days in position.
                num_days = max(i - entry_idx + 1, 1)  # so sum(daily_pnl) == net
                daily_dollar = net / num_days
                daily_ret = daily_dollar / notional
                for d in range(entry_idx, min(i + 1, n)):
                    daily_pnl.append({
                        "date": merged.iloc[d]["date"],
                        "daily_pnl": daily_dollar,
                        "daily_return": daily_ret,
                    })

                in_position = False

        i += 1

    # ── Build daily PnL DataFrame ──────────────────────────
    if daily_pnl:
        pnl_df = pd.DataFrame(daily_pnl)
        # Aggregate if multiple trades overlap (shouldn't happen here)
        pnl_df = pnl_df.groupby("date").sum().reset_index()
        pnl_df["cumulative_pnl"] = pnl_df["daily_pnl"].cumsum()
        pnl_df["cumulative_return"] = (1 + pnl_df["daily_return"]).cumprod() - 1
    else:
        pnl_df = pd.DataFrame(columns=["date", "daily_pnl", "daily_return",
                                        "cumulative_pnl", "cumulative_return"])

    n_stopped = sum(1 for t in trades if getattr(t, "stopped_out", False))
    print(f"[backtest] Completed: {len(trades)} trades, "
          f"{len(pnl_df)} trading days with PnL.  "
          f"(notional=${notional:,.0f}, stop-loss={stop_loss_pct:.0%})")
    if trades:
        wins = sum(1 for t in trades if t.net_pnl > 0)
        total_pnl = sum(t.net_pnl for t in trades)
        avg_contracts = np.mean([t.n_contracts for t in trades])
        print(f"[backtest] Win rate: {wins}/{len(trades)} "
              f"({100*wins/len(trades):.1f}%)")
        if n_stopped > 0:
            print(f"[backtest] Stop-loss exits: {n_stopped}/{len(trades)}")
        print(f"[backtest] Total net PnL: ${total_pnl:,.2f}  "
              f"(avg {avg_contracts:.0f} contracts/trade)")

    return trades, pnl_df


# ────────────────────────────────────────────────────────────
# 4.  Convert Trades to DataFrame
# ────────────────────────────────────────────────────────────

def trades_to_dataframe(trades):
    """
    Convert a list of Trade objects to a pandas DataFrame.

    Parameters
    ----------
    trades : List[Trade]

    Returns
    -------
    pd.DataFrame
    """
    if not trades:
        return pd.DataFrame()

    records = []
    for t in trades:
        records.append({
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "direction": t.direction,
            "direction_label": "short_vol" if t.direction == 1 else "long_vol",
            "n_contracts": t.n_contracts,
            "entry_iv": t.entry_iv,
            "realised_vol": t.realised_vol,
            "iv_rv_spread": t.entry_iv - t.realised_vol,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "straddle_premium": t.straddle_premium,
            "option_pnl": t.option_pnl,
            "hedge_pnl": t.hedge_pnl,
            "txn_cost": t.txn_cost,
            "net_pnl": t.net_pnl,
            "holding_days": t.holding_days,
            "stopped_out": getattr(t, "stopped_out", False),
        })
    return pd.DataFrame(records)
