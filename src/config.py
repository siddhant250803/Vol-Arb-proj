"""
Central configuration for the IV vs Forecast RV project.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Group 4 MS&E244"
OPTIONS_DIR = DATA_DIR / "Options"
RATES_DIR = DATA_DIR / "Risk-Free"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
REPORTS_DIR = OUTPUT_DIR / "reports"

OPTIONS_FILE = OPTIONS_DIR / "spx-weeklies_daily_friday-expiration_all.csv"
OPTIONS_FILTERED_FILE = OPTIONS_DIR / "spx-weeklies-filtered.csv"
YIELD_FILE = RATES_DIR / "yield_panel_daily_frequency_monthly_maturity.csv"

OPTIONS_USECOLS = [
    "date", "exdate", "cp_flag", "strike_price",
    "best_bid", "best_offer", "volume", "open_interest",
    "impl_volatility", "delta", "gamma", "vega", "theta",
    "forward_price", "exercise_style",
]

COL = dict(
    secid="secid",
    date="date",
    exdate="exdate",
    cp_flag="cp_flag",
    strike_raw="strike_price",       # raw column (×1000)
    bid="best_bid",
    ask="best_offer",
    volume="volume",
    open_interest="open_interest",
    iv="impl_volatility",
    delta="delta",
    gamma="gamma",
    vega="vega",
    theta="theta",
    forward="forward_price",
    spot="spx_price",
    exercise="exercise_style",
)

STRIKE_DIVISOR = 1000               # OptionMetrics stores strike × 1000
MIN_BID = 0.05                      # drop quotes with bid < $0.05
MAX_SPREAD_RATIO = 1.0              # drop if (ask-bid)/mid > 100 %
MIN_DTE = 7                         # minimum days-to-expiration
MAX_DTE = 60                        # maximum days-to-expiration
MONEYNESS_BAND = 0.20               # keep strikes within ±20% of spot

ATM_DELTA_BAND = 0.10               # |delta| ∈ [0.40, 0.60] for ATM
CONSTANT_MATURITY_DAYS = 30          # target constant-maturity tenor
VARIANCE_SWAP_STRIKE_BAND = 0.05     # ±5% for variance swap
TRADING_DAYS_PER_YEAR = 252

RV_HORIZONS = {                      # look-back windows (trading days)
    "daily": 1,
    "weekly": 5,
    "monthly": 22,
}
RV_FORECAST_HORIZON = 22             # forward-looking forecast (≈ 1 month)
ANNUALISATION_FACTOR = 252           # for annualising daily variance

HAR_LAGS = [1, 5, 22]               # daily, weekly, monthly
GARCH_P, GARCH_Q = 1, 1             # GARCH(1,1) default order

NOTIONAL_CAPITAL = 1_000_000         # total capital allocated ($)
CONTRACT_MULTIPLIER = 100            # SPX option multiplier (×100)
SIGNAL_ZSCORE_ENTRY = 1.0            # enter when |z| > threshold
SIGNAL_LOOKBACK = 252                # rolling window for z-score
HEDGE_FREQUENCY = "daily"            # delta-hedge frequency
TRANSACTION_COST_BPS = 5             # one-way cost in basis points
POSITION_HOLD_DAYS = 5               # max hold for Friday-expiring weeklies; capped at days to next Fri
MAX_HOLD_DAYS_WEEKLIES = 5           # do not test hold_days > this in param sweeps (options expire weekly)
STOP_LOSS_PCT = 0.25                 # exit when unrealized loss reaches 25% of that trade's value

FIGURE_DPI = 150
FIGURE_SIZE = (14, 7)
STYLE = "seaborn-v0_8-whitegrid"

PLOT_COLORS = {
    "50": "#FCE9E9",
    "100": "#F6C1C1",
    "200": "#F09999",
    "300": "#EA7171",
    "400": "#E44949",
    "500": "#DE2121",   # primary
    "600": "#B61B1B",
    "700": "#8C1515",
    "800": "#660F0F",
    "900": "#3E0909",
    "950": "#160303",
}
PLOT_PALETTE = [PLOT_COLORS[k] for k in ("200", "400", "600", "800", "300", "700", "500", "900")]
PLOT_PRIMARY = PLOT_COLORS["500"]
PLOT_SECONDARY = PLOT_COLORS["700"]
PLOT_LIGHT = PLOT_COLORS["100"]
PLOT_ACCENT = PLOT_COLORS["400"]
PLOT_NEUTRAL = "#4a4a4a"
PLOT_POSITIVE = "#2d7d2d"   # green for long vol / positive
PLOT_NEGATIVE = PLOT_COLORS["600"]   # for short vol / negative

FOMC_DATES = [
    # 1996
    "1996-01-30", "1996-03-26", "1996-05-21", "1996-07-02",
    "1996-08-20", "1996-09-24", "1996-11-13", "1996-12-17",
    # 1997
    "1997-02-04", "1997-03-25", "1997-05-20", "1997-07-01",
    "1997-08-19", "1997-09-30", "1997-11-12", "1997-12-16",
    # 1998
    "1998-02-03", "1998-03-31", "1998-05-19", "1998-06-30",
    "1998-08-18", "1998-09-29", "1998-11-17", "1998-12-22",
    # 1999
    "1999-02-02", "1999-03-30", "1999-05-18", "1999-06-29",
    "1999-08-24", "1999-10-05", "1999-11-16", "1999-12-21",
    # 2000
    "2000-02-01", "2000-03-21", "2000-05-16", "2000-06-27",
    "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-30", "2001-03-20", "2001-05-15", "2001-06-26",
    "2001-08-21", "2001-10-02", "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-29", "2002-03-19", "2002-05-07", "2002-06-25",
    "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-28", "2003-03-18", "2003-05-06", "2003-06-24",
    "2003-08-12", "2003-09-15", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-27", "2004-03-16", "2004-05-04", "2004-06-29",
    "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-01", "2005-03-22", "2005-05-03", "2005-06-29",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-27", "2006-05-10", "2006-06-28",
    "2006-08-08", "2006-09-20", "2006-10-24", "2006-12-12",
    # 2007
    "2007-01-30", "2007-03-20", "2007-05-09", "2007-06-27",
    "2007-08-07", "2007-09-18", "2007-10-30", "2007-12-11",
    # 2008
    "2008-01-29", "2008-03-18", "2008-04-29", "2008-06-24",
    "2008-08-05", "2008-09-16", "2008-10-28", "2008-12-15",
    # 2009
    "2009-01-27", "2009-03-17", "2009-04-28", "2009-06-23",
    "2009-08-11", "2009-09-22", "2009-11-03", "2009-12-15",
    # 2010
    "2010-01-26", "2010-03-16", "2010-04-27", "2010-06-22",
    "2010-08-10", "2010-09-21", "2010-11-02", "2010-12-14",
    # 2011
    "2011-01-25", "2011-03-15", "2011-04-26", "2011-06-21",
    "2011-08-09", "2011-09-20", "2011-11-01", "2011-12-13",
    # 2012
    "2012-01-24", "2012-03-13", "2012-04-24", "2012-06-19",
    "2012-07-31", "2012-09-12", "2012-10-23", "2012-12-11",
    # 2013
    "2013-01-29", "2013-03-19", "2013-04-30", "2013-06-18",
    "2013-07-30", "2013-09-17", "2013-10-29", "2013-12-17",
    # 2014
    "2014-01-28", "2014-03-18", "2014-04-29", "2014-06-17",
    "2014-07-29", "2014-09-16", "2014-10-28", "2014-12-16",
    # 2015
    "2015-01-27", "2015-03-17", "2015-04-28", "2015-06-16",
    "2015-07-28", "2015-09-16", "2015-10-27", "2015-12-15",
    # 2016
    "2016-01-26", "2016-03-15", "2016-04-26", "2016-06-14",
    "2016-07-26", "2016-09-20", "2016-11-01", "2016-12-13",
    # 2017
    "2017-01-31", "2017-03-14", "2017-05-02", "2017-06-13",
    "2017-07-25", "2017-09-19", "2017-10-31", "2017-12-12",
    # 2018
    "2018-01-30", "2018-03-20", "2018-05-01", "2018-06-12",
    "2018-07-31", "2018-09-25", "2018-11-07", "2018-12-18",
    # 2019
    "2019-01-29", "2019-03-19", "2019-04-30", "2019-06-18",
    "2019-07-30", "2019-09-17", "2019-10-29", "2019-12-10",
    # 2020
    "2020-01-28", "2020-03-15", "2020-04-28", "2020-06-09",
    "2020-07-28", "2020-09-15", "2020-11-04", "2020-12-15",
    # 2021
    "2021-01-26", "2021-03-16", "2021-04-27", "2021-06-15",
    "2021-07-27", "2021-09-21", "2021-11-02", "2021-12-14",
    # 2022
    "2022-01-25", "2022-03-15", "2022-05-03", "2022-06-14",
    "2022-07-26", "2022-09-20", "2022-11-01", "2022-12-13",
    # 2023
    "2023-01-31", "2023-03-21", "2023-05-02", "2023-06-13",
    "2023-07-25", "2023-09-19", "2023-10-31", "2023-12-12",
    # 2024
    "2024-01-30", "2024-03-19", "2024-04-30", "2024-06-11",
    "2024-07-30", "2024-09-17", "2024-11-06", "2024-12-17",
    # 2025
    "2025-01-28", "2025-03-18", "2025-05-06", "2025-06-17",
]
