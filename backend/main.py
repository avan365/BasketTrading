"""
FastAPI backend for Basket Trading with Bayesian Optimization.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core import (
    DataFetcher,
    DEFAULT_BASKETS,
    BasketTradingEngine,
    TradingStrategy,
    BayesianStrategyOptimizer,
    MultiObjectiveOptimizer,
    run_optimization_pipeline
)

# Initialize FastAPI app
app = FastAPI(
    title="Basket Trading Optimizer",
    description="ML-powered basket trading optimization using Bayesian methods",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
data_fetcher = DataFetcher()
executor = ThreadPoolExecutor(max_workers=4)
optimization_jobs: Dict[str, Dict] = {}


# ============== Pydantic Models ==============

class TickerList(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class OptimizationRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    benchmark: str = "SPY"
    objective: str = "sharpe"  # sharpe, sortino, calmar, return, risk_adjusted
    n_iterations: int = Field(default=50, ge=10, le=200)
    optimization_method: str = "bayesian"  # bayesian, multi_objective
    initial_capital: float = Field(default=100000, ge=1000)


class BacktestRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None
    initial_capital: float = 100000


class WeightOptimizationRequest(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    method: str = "mean_variance"  # mean_variance, risk_parity, momentum, min_variance
    max_weight: float = 0.25
    min_weight: float = 0.0


# ============== Helper Functions ==============

def get_date_range(start_date: Optional[str], end_date: Optional[str]):
    """Get default date range if not specified."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    return start_date, end_date


def run_optimization_sync(job_id: str, request: OptimizationRequest):
    """Run optimization synchronously in background."""
    try:
        optimization_jobs[job_id]['status'] = 'running'
        
        start_date, end_date = get_date_range(request.start_date, request.end_date)
        
        # Fetch data
        basket_data = data_fetcher.fetch_basket_data(
            request.tickers, start_date, end_date
        )
        
        if len(basket_data) < 2:
            optimization_jobs[job_id]['status'] = 'failed'
            optimization_jobs[job_id]['error'] = 'Could not fetch sufficient data'
            return
        
        prices = data_fetcher.get_aligned_prices(basket_data)
        
        # Fetch benchmark
        benchmark_data = data_fetcher.fetch_single_stock(
            request.benchmark, start_date, end_date
        )
        benchmark_returns = benchmark_data['returns'] if not benchmark_data.empty else None
        
        # Run optimization
        result = run_optimization_pipeline(
            prices=prices,
            benchmark_returns=benchmark_returns,
            optimization_method=request.optimization_method,
            objective=request.objective,
            n_iterations=request.n_iterations
        )
        
        optimization_jobs[job_id]['status'] = 'completed'
        optimization_jobs[job_id]['result'] = result
        optimization_jobs[job_id]['tickers'] = request.tickers
        
    except Exception as e:
        optimization_jobs[job_id]['status'] = 'failed'
        optimization_jobs[job_id]['error'] = str(e)


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Basket Trading Optimizer API",
        "version": "1.0.0",
        "endpoints": [
            "/api/baskets",
            "/api/data/fetch",
            "/api/optimize/weights",
            "/api/optimize/strategy",
            "/api/backtest"
        ]
    }


@app.get("/api/baskets")
async def get_default_baskets():
    """Get predefined stock baskets."""
    return {
        "baskets": DEFAULT_BASKETS,
        "descriptions": {
            "tech_leaders": "Major technology companies",
            "finance": "Financial sector leaders",
            "healthcare": "Healthcare and pharmaceutical companies",
            "consumer": "Consumer goods and retail",
            "energy": "Energy sector companies",
            "diversified": "Diversified cross-sector portfolio"
        }
    }


@app.post("/api/data/fetch")
async def fetch_basket_data(request: TickerList):
    """Fetch historical data for a basket of stocks."""
    start_date, end_date = get_date_range(request.start_date, request.end_date)
    
    try:
        basket_data = data_fetcher.fetch_basket_data(
            request.tickers, start_date, end_date
        )
        
        if not basket_data:
            raise HTTPException(status_code=404, detail="No data found for tickers")
        
        # Get aligned prices
        prices = data_fetcher.get_aligned_prices(basket_data)
        returns = data_fetcher.get_aligned_returns(basket_data)
        
        # Calculate summary statistics
        stats = {}
        for ticker in basket_data:
            df = basket_data[ticker]
            if not df.empty:
                stats[ticker] = {
                    'current_price': round(df['close'].iloc[-1], 2),
                    'total_return': round((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100, 2),
                    'volatility': round(df['returns'].std() * np.sqrt(252) * 100, 2),
                    'avg_volume': int(df['volume'].mean()) if 'volume' in df.columns else 0
                }
        
        return {
            "tickers": list(basket_data.keys()),
            "date_range": {
                "start": start_date,
                "end": end_date,
                "trading_days": len(prices)
            },
            "statistics": stats,
            "correlation_matrix": returns.corr().round(3).to_dict(),
            "prices": {
                "dates": prices.index.strftime('%Y-%m-%d').tolist(),
                "data": {ticker: prices[ticker].tolist() for ticker in prices.columns}
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/weights")
async def optimize_weights(request: WeightOptimizationRequest):
    """Optimize portfolio weights using various methods."""
    start_date, end_date = get_date_range(request.start_date, request.end_date)
    
    try:
        basket_data = data_fetcher.fetch_basket_data(
            request.tickers, start_date, end_date
        )
        
        if len(basket_data) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid tickers")
        
        prices = data_fetcher.get_aligned_prices(basket_data)
        engine = BasketTradingEngine(prices)
        
        # Optimize based on method
        if request.method == "mean_variance":
            weights = engine.optimize_weights_mean_variance(
                max_weight=request.max_weight,
                min_weight=request.min_weight
            )
        elif request.method == "risk_parity":
            weights = engine.optimize_weights_risk_parity()
        elif request.method == "momentum":
            weights = engine.optimize_weights_momentum()
        elif request.method == "min_variance":
            weights = engine.optimize_weights_minimum_variance(
                max_weight=request.max_weight
            )
        else:
            weights = np.ones(len(request.tickers)) / len(request.tickers)
        
        # Calculate expected metrics
        exp_return = weights @ engine.mean_returns
        exp_vol = np.sqrt(weights @ engine.cov_matrix @ weights)
        exp_sharpe = (exp_return - 0.05) / exp_vol if exp_vol > 0 else 0
        
        return {
            "method": request.method,
            "tickers": list(prices.columns),
            "weights": {
                ticker: round(float(w), 4) 
                for ticker, w in zip(prices.columns, weights)
            },
            "expected_metrics": {
                "annual_return": round(exp_return * 100, 2),
                "volatility": round(exp_vol * 100, 2),
                "sharpe_ratio": round(exp_sharpe, 3)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/strategy")
async def start_strategy_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Start Bayesian optimization of trading strategy parameters."""
    job_id = str(uuid.uuid4())
    
    optimization_jobs[job_id] = {
        'status': 'pending',
        'created_at': datetime.now().isoformat(),
        'request': request.model_dump()
    }
    
    # Run optimization in background
    background_tasks.add_task(run_optimization_sync, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Optimization started. Poll /api/optimize/status/{job_id} for results."
    }


@app.get("/api/optimize/status/{job_id}")
async def get_optimization_status(job_id: str):
    """Get status of an optimization job."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = optimization_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job['status'],
        "created_at": job['created_at']
    }
    
    if job['status'] == 'completed':
        response['result'] = job['result']
        response['tickers'] = job['tickers']
    elif job['status'] == 'failed':
        response['error'] = job.get('error', 'Unknown error')
    
    return response


@app.post("/api/optimize/quick")
async def quick_optimization(request: OptimizationRequest):
    """Run a quick optimization (fewer iterations, synchronous)."""
    start_date, end_date = get_date_range(request.start_date, request.end_date)
    
    try:
        basket_data = data_fetcher.fetch_basket_data(
            request.tickers, start_date, end_date
        )
        
        if len(basket_data) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid tickers")
        
        prices = data_fetcher.get_aligned_prices(basket_data)
        
        benchmark_data = data_fetcher.fetch_single_stock(
            request.benchmark, start_date, end_date
        )
        benchmark_returns = benchmark_data['returns'] if not benchmark_data.empty else None
        
        # Run with fewer iterations for quick results
        result = run_optimization_pipeline(
            prices=prices,
            benchmark_returns=benchmark_returns,
            optimization_method=request.optimization_method,
            objective=request.objective,
            n_iterations=min(request.n_iterations, 30)  # Cap at 30 for quick mode
        )
        
        return {
            "tickers": request.tickers,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """Run a backtest with specified weights and strategy parameters."""
    start_date, end_date = get_date_range(request.start_date, request.end_date)
    
    if len(request.tickers) != len(request.weights):
        raise HTTPException(
            status_code=400, 
            detail="Number of tickers must match number of weights"
        )
    
    try:
        basket_data = data_fetcher.fetch_basket_data(
            request.tickers, start_date, end_date
        )
        
        if len(basket_data) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid tickers")
        
        prices = data_fetcher.get_aligned_prices(basket_data)
        
        # Normalize weights
        weights = np.array(request.weights)
        weights = weights / weights.sum()
        
        # Create strategy
        strategy_params = request.strategy_params or {}
        strategy = TradingStrategy(
            weights=weights,
            rebalance_frequency=strategy_params.get('rebalance_frequency', 21),
            rebalance_threshold=strategy_params.get('rebalance_threshold', 0.05),
            max_position_size=strategy_params.get('max_position_size', 0.25),
            min_position_size=strategy_params.get('min_position_size', 0.02),
            momentum_weight=strategy_params.get('momentum_weight', 0.3),
            vol_target=strategy_params.get('vol_target', 0.15),
            use_vol_scaling=strategy_params.get('use_vol_scaling', True)
        )
        
        # Run backtest
        engine = BasketTradingEngine(prices)
        results, metrics = engine.backtest_strategy(
            strategy, 
            initial_capital=request.initial_capital
        )
        
        # Get efficient frontier
        frontier = engine.get_efficient_frontier(n_portfolios=50)
        
        return {
            "tickers": list(prices.columns),
            "weights": {
                ticker: round(float(w), 4) 
                for ticker, w in zip(prices.columns, weights)
            },
            "metrics": metrics.to_dict(),
            "backtest_data": {
                "dates": results.index.strftime('%Y-%m-%d').tolist(),
                "portfolio_values": results['portfolio_value'].round(2).tolist(),
                "cumulative_returns": (results['cumulative_returns'] * 100).fillna(0).round(2).tolist(),
                "drawdown": (results['drawdown'] * 100).fillna(0).round(2).tolist()
            },
            "efficient_frontier": {
                "returns": frontier['return'].round(4).tolist(),
                "volatilities": frontier['volatility'].round(4).tolist(),
                "sharpe_ratios": frontier['sharpe'].round(4).tolist()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/correlation/{basket_name}")
async def get_basket_correlation(basket_name: str):
    """Get correlation analysis for a predefined basket."""
    if basket_name not in DEFAULT_BASKETS:
        raise HTTPException(status_code=404, detail="Basket not found")
    
    tickers = DEFAULT_BASKETS[basket_name]
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        basket_data = data_fetcher.fetch_basket_data(tickers, start_date, end_date)
        returns = data_fetcher.get_aligned_returns(basket_data)
        
        corr_matrix = returns.corr()
        
        return {
            "basket": basket_name,
            "tickers": list(corr_matrix.columns),
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "average_correlation": round(
                corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(), 
                3
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

