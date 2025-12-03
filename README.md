# 📈 Basket Trading Optimizer

A machine learning-powered application for optimizing basket trading strategies using **Bayesian Optimization**. This tool helps traders and portfolio managers find optimal parameters for their trading strategies through intelligent exploration of the parameter space.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-18-61dafb.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)

## ✨ Features

### 🎯 Portfolio Optimization Methods

- **Mean-Variance Optimization** - Classic Markowitz portfolio optimization
- **Risk Parity** - Equal risk contribution across assets
- **Minimum Variance** - Lowest volatility portfolio
- **Momentum-Based** - Allocation based on recent performance

### 🧠 Bayesian Optimization

- **Gaussian Process Regression** - Models the objective function
- **Expected Improvement Acquisition** - Intelligent next-point selection
- **Multi-objective Optimization** - Balance multiple metrics
- **Early Stopping** - Converges efficiently

### 📊 Optimized Parameters

- Rebalancing frequency and thresholds
- Position size limits (min/max weights)
- Stop-loss and take-profit levels
- Momentum window and weighting
- Volatility targeting and scaling

### 📈 Performance Metrics

- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown analysis
- Win Rate and Profit Factor
- Alpha and Beta vs benchmark
- Information Ratio

## 🏗️ Architecture

```
Basket Trading/
├── backend/
│   ├── core/
│   │   ├── data_fetcher.py      # Yahoo Finance data retrieval
│   │   ├── basket_engine.py     # Portfolio optimization engine
│   │   └── bayesian_optimizer.py # Bayesian optimization logic
│   └── main.py                   # FastAPI application
├── frontend/
│   └── src/
│       ├── App.jsx              # Main React application
│       └── index.css            # Tailwind styles
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+

### Backend Setup

```bash
# Navigate to project directory
cd "Basket Trading"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
# In a new terminal, navigate to frontend
cd "Basket Trading/frontend"

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at:

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📡 API Endpoints

### Data Endpoints

| Endpoint          | Method | Description                  |
| ----------------- | ------ | ---------------------------- |
| `/api/baskets`    | GET    | Get predefined stock baskets |
| `/api/data/fetch` | POST   | Fetch historical price data  |

### Optimization Endpoints

| Endpoint                    | Method | Description                        |
| --------------------------- | ------ | ---------------------------------- |
| `/api/optimize/weights`     | POST   | Optimize portfolio weights         |
| `/api/optimize/strategy`    | POST   | Start async Bayesian optimization  |
| `/api/optimize/quick`       | POST   | Run quick synchronous optimization |
| `/api/optimize/status/{id}` | GET    | Check optimization job status      |

### Analysis Endpoints

| Endpoint                             | Method | Description              |
| ------------------------------------ | ------ | ------------------------ |
| `/api/backtest`                      | POST   | Run strategy backtest    |
| `/api/analysis/correlation/{basket}` | GET    | Get correlation analysis |

## 🔧 Configuration

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

## 📊 Example Usage

### Python API

```python
from backend.core import (
    DataFetcher,
    BasketTradingEngine,
    BayesianStrategyOptimizer,
    run_optimization_pipeline
)

# Fetch data
fetcher = DataFetcher()
basket_data = fetcher.fetch_basket_data(
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    '2022-01-01',
    '2024-01-01'
)
prices = fetcher.get_aligned_prices(basket_data)

# Initialize engine
engine = BasketTradingEngine(prices)

# Get optimal weights
weights = engine.optimize_weights_mean_variance()

# Run Bayesian optimization
optimizer = BayesianStrategyOptimizer(
    engine=engine,
    base_weights=weights,
    objective='sharpe'
)
result = optimizer.optimize(n_calls=50)

print(f"Best Sharpe: {result.best_score:.4f}")
print(f"Best params: {result.best_params}")
```

### REST API

```bash
# Fetch data
curl -X POST http://localhost:8000/api/data/fetch \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL"]}'

# Run optimization
curl -X POST http://localhost:8000/api/optimize/quick \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "objective": "sharpe",
    "n_iterations": 30
  }'
```

## 🧪 How Bayesian Optimization Works

1. **Initialization**: Sample random points in the parameter space
2. **Surrogate Model**: Fit a Gaussian Process to observed points
3. **Acquisition**: Use Expected Improvement to select the next point
4. **Evaluation**: Run backtest with new parameters
5. **Update**: Add result to observations and repeat

The algorithm efficiently explores the parameter space, balancing:

- **Exploration**: Trying new regions
- **Exploitation**: Refining known good regions

## 📈 Pre-defined Baskets

| Basket         | Description          | Tickers                                   |
| -------------- | -------------------- | ----------------------------------------- |
| `tech_leaders` | Major tech companies | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA |
| `finance`      | Financial sector     | JPM, BAC, WFC, GS, MS, C, AXP             |
| `healthcare`   | Healthcare/Pharma    | JNJ, UNH, PFE, MRK, ABBV, LLY, TMO        |
| `consumer`     | Consumer goods       | WMT, PG, KO, PEP, COST, HD, MCD           |
| `energy`       | Energy sector        | XOM, CVX, COP, SLB, EOG, MPC, PSX         |
| `diversified`  | Cross-sector mix     | AAPL, JPM, JNJ, XOM, WMT, GOOGL, PG, UNH  |

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 📄 License

MIT License - feel free to use and modify for your own projects.

---

Built with 🧠 Bayesian ML and ☕ lots of coffee
