**Dumbed-down guide for a beginner trader**

### Basket

Choose a collection of stocks that you believe will perform well together. This can be based on industry, sector, or any other criteria that you believe will be correlated in the future.

### Dates

long-term robust strategy with consistent performance in bull AND bear markets --> 5 to 10 years of data
designed for today's market --> 2 to 3 years of data
trend-following of short-term strategy --> 6 to 18 months of data

### Portfolio Optimization Method (decides weight of each asset in the basket)

1. Mean-Variance (Markowitz) --> finds the best tradeoff between return and risk
2. Risk Parity --> equal risk contribution from each asset
3. Momentum --> gives more weight to assets that have performed well recently (higher risk but higher return potential)
4. Minimum Variance --> finds the portfolio with the lowest possible variance and volatility

### Optimization (auto-determined / fixed) --> gives insights

- rebalancing frequency = how often you should adjust your portfolio (e.g., every 10 days, 30 days, etc.)
- rebalance threshold = how far weights can drift before the system adjusts
- max/min position size = limits how much you can allocate to one stock
- stop-loss / take-profit = values that ensure automatic protection and profit capture
- momentum window = how far back you should look to detect a trend (10–60 days)
- Momentum weight = how strongly momentum influences your trades
- volatility target = annualized volatility level your strategy tries to maintain
- Volatility lookback = how many days of price history the strategy uses to measure volatility (10–60 days)
- use vol scaling = if true, your strategy automatically scales position size up or down to hit the volatility target. if false, your strategy will try to hit the volatility target exactly.

### Optimization Objective

1. Sharpe Ratio = (Expected Return - Risk-Free Rate) / Volatility [best for risk-adjusted returns]
2. Sortino Ratio = (Expected Return - Risk-Free Rate) / Downside Volatility [best for risk-adjusted returns, and more conservative than Sharpe Ratio]
3. Calmar Ratio = (Expected Return - Risk-Free Rate) / Maximum Drawdown [best for risk-adjusted returns to avoid big losses]
4. Annualized Return = (Final Portfolio Value - Initial Portfolio Value) / Initial Portfolio Value [maximimze raw performance regardless of risk]
5. Risk-Adjusted Return = (Expected Return - Risk-Free Rate) / Volatility [best for balanced performance]

### Performance Metrics

- Sharpe / Sortino / Calmar - tells you whether returns are worth the risk
- Max Drawdown - shows the worst historical drop, which is crucial for realistic expectation setting
- Win Rate / Profit Factor - useful for evaluating systems with frequent trading
- Alpha / Beta - (alpha = how much you beat the market) & (beta = how much you move with the market)
- Information Ratio = consistency of outperformance
