# BasketTrading
Description: Improve Basket Trading using Beysian Optimization

**1. What is Basket Trading?**
Basket trading is an *invest or trading strategy* where multiple securities are bought/sold together as a single unit (aka the basket). Instead of trading individual stocks or assets one by one, the trader executes one order that includes a group of assets.
Basket Types:
* Sector Basket - specific industries (banking, tech, etc)
* Index Basket - indices like S&P 500 or NIFTY 50
* Currency Basket - FX markets (USD or even EUR/JPY/GBP mix)
* **Algo Basket** - Selected based on quantitative signals or models
* Arbitrage Basket - pair-trading or market-neutral strategies

Benefits posed by basket trading include diversification (since it reduces risk by spreading exposure across multiple assets), efficiency (it allows bulk execution instead of individual orders), trategy alignment (useful for thematic trading) and hedging (can balance long and short positions within the same basket).

EXAMPLE:
Sector Basket --> Tech Sector = { AAPL, MSFT, AMZN, GOOGL, NVDA }

general steps:
define your basket's theme --> select basket constituents --> allocate weights [equal weight or weighted by market cap, volatility, etc] --> execute orders
* monitor performance over time and rebalance weights / constituents if needed


**2. What is Beysian Optimization?**
Essentially, Beysian Optimization is a strategy for optimizing functions that are expensive/slow/difficult to evaluate. In this case, Bayesian Optimization can be used to select assets within a basket, allocate weights or even tune the parameters for the trading strategy applied to the basket. We will come back to this later.

So how does it work? Instead of brute-forcing or randomly searching, it selects the next point to test based on probability (which means that it can find the best solution in fewer trials).
On a more technical base, it first tests a few random choices and then builds a surrogate model (possibly Gaussian Process) to approximate the function. A acquisition function then decides where to next test (the hard part is the challenge of balancing between exploring unknown regions vs exploiting promising regions). Upon evaluating the real function at this new suggested point, the model is update. This process is repeated until the function is satisfactory (loop till success).

Surrogate Model: approximates the unknown function cheaply (GP, Random Forests, TPE)
Acquisition Function: intelligently selects the next best point
Objective Function: the thing that we are trying to optimise (the end-product of a sort)

Alternative Models (that we are not using in this project):
1. Grid Search
2. Random Search

**3. Bayesian Optimization in Basket Trading**
1. Asset Selection
when asset pool is large, Bayesian Optimization can help choose the subset that performs best under constraints

2. Weight allocation
instead of assigning equal weights or manually allocating weights, Bayesian Optimization can search for the best weight combination to maximize chosen objective. possible objectives include returns, sharpe ratio, risk-adjusted return, min drawdown, etc...

3. Parameter Tuning for Trading Strategies Applied [what BasketTrading focuses on]
is applying an automated or algorithmic strategy to a basket, there may parameters (e.g: lookback window, stop-loss distance, rebalance frequency, etc) & Bayesian Optimization helps to find optimal values faster.







