**Basket Trading Optimizer**

A machine learning-powered application for optimizing basket trading strategies using **Bayesian Optimization**. This tool helps traders and portfolio managers find optimal parameters for their trading strategies through intelligent exploration of the parameter space.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-18-61dafb.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)

## Features

### Portfolio Optimization Methods

- **Mean-Variance Optimization** - Classic Markowitz portfolio optimization
- **Risk Parity** - Equal risk contribution across assets
- **Minimum Variance** - Lowest volatility portfolio
- **Momentum-Based** - Allocation based on recent performance

### Bayesian Optimization

- **Gaussian Process Regression** - Models the objective function
- **Expected Improvement Acquisition** - Intelligent next-point selection
- **Multi-objective Optimization** - Balance multiple metrics
- **Early Stopping** - Converges efficiently

### Optimized Parameters

- Rebalancing frequency and thresholds
- Position size limits (min/max weights)
- Stop-loss and take-profit levels
- Momentum window and weighting
- Volatility targeting and scaling

### Performance Metrics

- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown analysis
- Win Rate and Profit Factor
- Alpha and Beta vs benchmark
- Information Ratio

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+

### Backend Setup

```bash
cd "Basket Trading"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd "Basket Trading/frontend"
npm install
npm run dev
```

## Configuration

### Optimization Parameters

```python
# Search space for Bayesian optimization
SEARCH_SPACE = {
    'rebalance_frequency': (5, 63),      # Days
    'rebalance_threshold': (0.01, 0.15), # Drift %
    'max_position_size': (0.15, 0.40),   # Max weight
    'min_position_size': (0.01, 0.10),   # Min weight
    'stop_loss': (0.05, 0.30),           # Stop loss %
    'take_profit': (0.10, 0.50),         # Take profit %
    'momentum_window': (10, 60),         # Days
    'momentum_weight': (0.0, 0.6),       # Factor
    'vol_target': (0.08, 0.25),          # Target vol
    'vol_lookback': (10, 42),            # Days
}
```

### Optimization Objectives

- `sharpe` - Maximize Sharpe ratio (risk-adjusted return)
- `sortino` - Maximize Sortino ratio (downside risk-adjusted)
- `calmar` - Maximize Calmar ratio (return / max drawdown)
- `return` - Maximize annualized return
- `risk_adjusted` - Combined Sharpe + Sortino

## How Bayesian Optimization Works

1. **Initialization**: Sample random points in the parameter space
2. **Surrogate Model**: Fit a Gaussian Process to observed points
3. **Acquisition**: Use Expected Improvement to select the next point
4. **Evaluation**: Run backtest with new parameters
5. **Update**: Add result to observations and repeat

The algorithm efficiently explores the parameter space, balancing:

- **Exploration**: Trying new regions
- **Exploitation**: Refining known good regions

## Pre-defined Baskets

| Basket         | Description          | Tickers                                   |
| -------------- | -------------------- | ----------------------------------------- |
| `tech_leaders` | Major tech companies | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA |
| `finance`      | Financial sector     | JPM, BAC, WFC, GS, MS, C, AXP             |
| `healthcare`   | Healthcare/Pharma    | JNJ, UNH, PFE, MRK, ABBV, LLY, TMO        |
| `consumer`     | Consumer goods       | WMT, PG, KO, PEP, COST, HD, MCD           |
| `energy`       | Energy sector        | XOM, CVX, COP, SLB, EOG, MPC, PSX         |
| `diversified`  | Cross-sector mix     | AAPL, JPM, JNJ, XOM, WMT, GOOGL, PG, UNH  |
