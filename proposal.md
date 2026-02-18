Project Proposal

We want to test whether options’ implied volatility is systematically rich/cheap relative to what actually realises, and whether that gap is tradeable after realistic costs. The main focus is a clean “IV vs forecast RV” signal; a secondary extension is relative-value across strikes/maturities (surface mispricing), where the signal comes from deviations in skew/term structure rather than just the ATM level.
Prior research shows that model-free implied variance from option prices typically exceeds realized variance, reflecting a variance risk premium with predictive content. Advances in realized-volatility measurement and forecasting, such as HAR-RV and realized-GARCH models, enable systematic trading on the IV–RV spread, while studies of risk-neutral distributions and volatility-surface structure motivate relative-value and distribution-based volatility strategies.

Data: Options chains with accurate greeks/IV, realized volatility measures, corporate actions, dividends, borrow rates, transaction costs, risk free rate (US)
Initially we will look at SPX options, but can widen the number of stocks once we find some results

Method to generate a signal:
Obtain IV from ATM options
Forecast RV using HAR/ARCH/GARCH type models
Include provisions for known high-volatility events like FOMC events, NFP unemployment rate etc.
Core Signal: S = IV - RV; s* = norm(S)
Variance Swap model: Instead of comparing volatility using IV, build out a variance swap using IV of +/- 5% of SPX options, use the following discretized verson to get the implied vol.
Distribution-based. Instead of comparing single vol numbers, we forecast the future realised price distribution and compare it to the option-implied risk-neutral distribution extracted from the surface. One way to do this is to discretise standardized prices into bins and use logistic regression to estimate the probability of landing in each bin. The discrepancy between the forecast “realised” distribution and the implied distribution becomes a trading signal, especially for skew/tail mispricing.

Backtesting will implement delta-hedged option trades (e.g. ATM straddles/strangles) to isolate volatility exposure. Positions will be entered when the signal is extreme, delta-hedged at a fixed frequency, and exited at a fixed horizon or rolled by predefined rules. We will report net performance (Sharpe, cumulative P&L, drawdowns, tail losses), run portfolio sorts by signal strength, and stress-test robustness across horizons, maturities, hedging frequency, cost assumptions, and event windows/regimes.

References
Breeden, D. T., & Litzenberger, R. H. (1978). Prices of State-Contingent Claims Implicit in Option Prices.
Carr, P., & Madan, D. (1998). Towards a Theory of Volatility Trading.
Demeterfi, K., Derman, E., Kamal, M., & Zou, J. (1999). More Than You Ever Wanted to Know About Volatility Swaps.
Bakshi, G., Kapadia, N., & Madan, D. (2003). Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options.
Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and Forecasting Realized Volatility.
Bondarenko, O. (2003). Why Are Put Options So Expensive?
Gatheral, J. (2006). The Volatility Surface: A Practitioner’s Guide.
Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia.
Carr, P., & Wu, L. (2009). Variance Risk Premia.
Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility (HAR-RV).
Drechsler, I., & Yaron, A. (2011). What’s Vol Got to Do with It?
Hansen, P. R., Huang, Z., & Shek, H. H. (2012). Realized GARCH.
