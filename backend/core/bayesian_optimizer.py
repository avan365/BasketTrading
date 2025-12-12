"""
Bayesian Optimization module for optimizing basket trading strategy parameters.

Uses scikit-optimize's Gaussian Process-based Bayesian optimization to find
optimal hyperparameters for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import DeltaYStopper
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

from .basket_engine import BasketTradingEngine, TradingStrategy, PortfolioMetrics

warnings.filterwarnings('ignore')


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict]
    convergence_data: List[float]
    total_iterations: int
    computation_time: float
    best_metrics: Optional[PortfolioMetrics] = None
    
    def to_dict(self) -> Dict:
        return {
            'best_params': convert_numpy_types(self.best_params),
            'best_score': round(float(self.best_score), 6),
            'optimization_history': convert_numpy_types(self.optimization_history),
            'convergence_data': [round(float(x), 6) for x in self.convergence_data],
            'total_iterations': int(self.total_iterations),
            'computation_time': round(float(self.computation_time), 2),
            'best_metrics': self.best_metrics.to_dict() if self.best_metrics else None
        }


class BayesianStrategyOptimizer:
    """
    Bayesian optimization for trading strategy parameters.
    
    Uses Gaussian Process regression to model the objective function
    and Expected Improvement acquisition to select next evaluation points.
    """
    
    # Define the search space for strategy parameters
    SEARCH_SPACE = [
        Integer(5, 63, name='rebalance_frequency'),      # Days between rebalancing
        Real(0.01, 0.15, name='rebalance_threshold'),    # Drift threshold
        Real(0.15, 0.40, name='max_position_size'),      # Max single position
        Real(0.01, 0.10, name='min_position_size'),      # Min single position
        Real(0.05, 0.30, name='stop_loss'),              # Stop loss %
        Real(0.10, 0.50, name='take_profit'),            # Take profit %
        Integer(10, 60, name='momentum_window'),         # Momentum lookback
        Real(0.0, 0.6, name='momentum_weight'),          # Momentum factor
        Real(0.08, 0.25, name='vol_target'),             # Target volatility
        Integer(10, 42, name='vol_lookback'),            # Volatility lookback
        Categorical([True, False], name='use_vol_scaling')  # Use vol scaling
    ]
    
    PARAM_NAMES = [
        'rebalance_frequency', 'rebalance_threshold', 'max_position_size',
        'min_position_size', 'stop_loss', 'take_profit', 'momentum_window',
        'momentum_weight', 'vol_target', 'vol_lookback', 'use_vol_scaling'
    ]
    
    def __init__(
        self,
        engine: BasketTradingEngine,
        base_weights: np.ndarray,
        objective: str = 'sharpe',
        n_initial_points: int = 10
    ):
        """
        Initialize the optimizer.
        
        Args:
            engine: The basket trading engine with price data
            base_weights: Initial portfolio weights to optimize around
            objective: Optimization objective ('sharpe', 'sortino', 'calmar', 'return')
            n_initial_points: Number of random points before Bayesian optimization
        """
        self.engine = engine
        self.base_weights = base_weights
        self.objective = objective
        self.n_initial_points = n_initial_points
        
        # Track optimization history
        self.history: List[Dict] = []
        self.best_score = -np.inf
        self.best_params = None
        self.iteration = 0
    
    def _create_strategy(self, params: Dict) -> TradingStrategy:
        """Create a TradingStrategy from parameter dictionary."""
        return TradingStrategy(
            weights=self.base_weights.copy(),
            rebalance_frequency=int(params['rebalance_frequency']),
            rebalance_threshold=params['rebalance_threshold'],
            max_position_size=params['max_position_size'],
            min_position_size=params['min_position_size'],
            stop_loss=params['stop_loss'],
            take_profit=params['take_profit'],
            momentum_window=int(params['momentum_window']),
            momentum_weight=params['momentum_weight'],
            vol_target=params['vol_target'],
            vol_lookback=int(params['vol_lookback']),
            use_vol_scaling=params['use_vol_scaling']
        )
    
    def _objective_function(self, params_list: List) -> float:
        """
        Objective function to minimize (returns negative of target metric).
        
        Args:
            params_list: List of parameter values in order of SEARCH_SPACE
        
        Returns:
            Negative of the objective metric (for minimization)
        """
        # Convert list to dict
        params = dict(zip(self.PARAM_NAMES, params_list))
        
        # Validate constraints
        if params['min_position_size'] >= params['max_position_size']:
            return 1000  # Penalty for invalid configuration
        
        if params['stop_loss'] >= params['take_profit']:
            return 1000  # Penalty
        
        try:
            # Create and backtest strategy
            strategy = self._create_strategy(params)
            results, metrics = self.engine.backtest_strategy(strategy)
            
            # Get objective value
            if self.objective == 'sharpe':
                score = metrics.sharpe_ratio
            elif self.objective == 'sortino':
                score = metrics.sortino_ratio
            elif self.objective == 'calmar':
                score = metrics.calmar_ratio
            elif self.objective == 'return':
                score = metrics.annualized_return
            elif self.objective == 'risk_adjusted':
                # Custom: combination of Sharpe and Sortino
                score = 0.5 * metrics.sharpe_ratio + 0.5 * metrics.sortino_ratio
            else:
                score = metrics.sharpe_ratio
            
            # Track history
            self.iteration += 1
            record = {
                'iteration': self.iteration,
                'params': params.copy(),
                'score': score,
                'metrics': metrics.to_dict()
            }
            self.history.append(record)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.best_metrics = metrics
            
            # Return negative for minimization
            return -score
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1000  # Penalty for failed evaluation
    
    def optimize(
        self,
        n_calls: int = 50,
        method: str = 'gp',
        random_state: int = 42,
        verbose: bool = True,
        early_stopping_rounds: int = 15,
        early_stopping_delta: float = 0.001
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            n_calls: Total number of objective function evaluations
            method: Optimization method ('gp', 'forest', 'gbrt')
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
            early_stopping_rounds: Stop if no improvement for this many rounds
            early_stopping_delta: Minimum improvement to reset early stopping
        
        Returns:
            OptimizationResult with optimal parameters and history
        """
        start_time = time.time()
        
        # Reset state
        self.history = []
        self.best_score = -np.inf
        self.best_params = None
        self.iteration = 0
        
        # Select optimization method
        if method == 'gp':
            minimize_func = gp_minimize
        elif method == 'forest':
            minimize_func = forest_minimize
        elif method == 'gbrt':
            minimize_func = gbrt_minimize
        else:
            minimize_func = gp_minimize
        
        # Early stopping callback
        early_stopper = DeltaYStopper(
            delta=early_stopping_delta,
            n_best=early_stopping_rounds
        )
        
        # Run optimization
        if verbose:
            print(f"Starting Bayesian optimization with {n_calls} iterations...")
            print(f"Objective: maximize {self.objective}")
        
        result = minimize_func(
            func=self._objective_function,
            dimensions=self.SEARCH_SPACE,
            n_calls=n_calls,
            n_initial_points=self.n_initial_points,
            random_state=random_state,
            verbose=verbose,
            callback=[early_stopper] if early_stopping_rounds > 0 else None,
            acq_func='EI',  # Expected Improvement
            n_jobs=1  # Sequential for reproducibility
        )
        
        computation_time = time.time() - start_time
        
        # Extract convergence data
        convergence = []
        best_so_far = np.inf
        for y in result.func_vals:
            best_so_far = min(best_so_far, y)
            convergence.append(-best_so_far)  # Convert back to positive
        
        # Prepare result
        best_params = dict(zip(self.PARAM_NAMES, result.x))
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best {self.objective}: {-result.fun:.4f}")
            print(f"Time elapsed: {computation_time:.1f}s")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=-result.fun,
            optimization_history=self.history,
            convergence_data=convergence,
            total_iterations=len(result.func_vals),
            computation_time=computation_time,
            best_metrics=self.best_metrics if hasattr(self, 'best_metrics') else None
        )
    
    def optimize_weights_bayesian(
        self,
        n_calls: int = 30,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Use Bayesian optimization to find optimal portfolio weights directly.
        
        Returns:
            Optimized weight vector
        """
        n_assets = self.engine.n_assets
        
        # Create weight search space (bounded to sum approximately to 1)
        weight_space = [Real(0.02, 0.5, name=f'w_{i}') for i in range(n_assets)]
        
        def weight_objective(weights):
            # Normalize weights to sum to 1
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Create strategy with these weights
            strategy = TradingStrategy(weights=weights)
            _, metrics = self.engine.backtest_strategy(strategy)
            
            return -metrics.sharpe_ratio
        
        result = gp_minimize(
            func=weight_objective,
            dimensions=weight_space,
            n_calls=n_calls,
            random_state=random_state,
            acq_func='EI'
        )
        
        # Normalize final weights
        optimal_weights = np.array(result.x)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        return optimal_weights


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization balancing multiple trading metrics.
    """
    
    def __init__(
        self,
        engine: BasketTradingEngine,
        base_weights: np.ndarray,
        objectives: List[str] = ['sharpe', 'sortino', 'max_drawdown'],
        weights: Optional[List[float]] = None
    ):
        self.engine = engine
        self.base_weights = base_weights
        self.objectives = objectives
        self.objective_weights = weights or [1.0 / len(objectives)] * len(objectives)
    
    def _combined_objective(self, params_list: List) -> float:
        """Combined weighted objective function."""
        param_names = BayesianStrategyOptimizer.PARAM_NAMES
        params = dict(zip(param_names, params_list))
        
        # Validation
        if params['min_position_size'] >= params['max_position_size']:
            return 1000
        if params['stop_loss'] >= params['take_profit']:
            return 1000
        
        try:
            strategy = TradingStrategy(
                weights=self.base_weights.copy(),
                rebalance_frequency=int(params['rebalance_frequency']),
                rebalance_threshold=params['rebalance_threshold'],
                max_position_size=params['max_position_size'],
                min_position_size=params['min_position_size'],
                stop_loss=params['stop_loss'],
                take_profit=params['take_profit'],
                momentum_window=int(params['momentum_window']),
                momentum_weight=params['momentum_weight'],
                vol_target=params['vol_target'],
                vol_lookback=int(params['vol_lookback']),
                use_vol_scaling=params['use_vol_scaling']
            )
            
            _, metrics = self.engine.backtest_strategy(strategy)
            
            # Calculate combined score
            scores = []
            for obj, weight in zip(self.objectives, self.objective_weights):
                if obj == 'sharpe':
                    scores.append(metrics.sharpe_ratio * weight)
                elif obj == 'sortino':
                    scores.append(metrics.sortino_ratio * weight)
                elif obj == 'calmar':
                    scores.append(metrics.calmar_ratio * weight)
                elif obj == 'return':
                    scores.append(metrics.annualized_return * weight)
                elif obj == 'max_drawdown':
                    # Lower drawdown is better, so invert
                    scores.append((1 - metrics.max_drawdown) * weight)
                elif obj == 'volatility':
                    # Lower volatility is better, so invert (capped at 50%)
                    scores.append((0.5 - metrics.volatility) * weight)
            
            return -sum(scores)
            
        except Exception:
            return 1000
    
    def optimize(self, n_calls: int = 50, random_state: int = 42) -> OptimizationResult:
        """Run multi-objective optimization."""
        start_time = time.time()
        
        result = gp_minimize(
            func=self._combined_objective,
            dimensions=BayesianStrategyOptimizer.SEARCH_SPACE,
            n_calls=n_calls,
            n_initial_points=10,
            random_state=random_state,
            acq_func='EI'
        )
        
        computation_time = time.time() - start_time
        
        best_params = dict(zip(BayesianStrategyOptimizer.PARAM_NAMES, result.x))
        convergence = [-min(result.func_vals[:i+1]) for i in range(len(result.func_vals))]
        
        return OptimizationResult(
            best_params=best_params,
            best_score=-result.fun,
            optimization_history=[],
            convergence_data=convergence,
            total_iterations=len(result.func_vals),
            computation_time=computation_time
        )


def run_optimization_pipeline(
    prices: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    optimization_method: str = 'bayesian',
    objective: str = 'sharpe',
    n_iterations: int = 50,
    weight_method: str = 'mean_variance'
) -> Dict:
    """
    Complete optimization pipeline.
    
    Args:
        prices: DataFrame of asset prices
        benchmark_returns: Optional benchmark returns series
        optimization_method: 'bayesian', 'multi_objective', or 'grid'
        objective: Optimization target metric
        n_iterations: Number of optimization iterations
        weight_method: Initial weight optimization method
    
    Returns:
        Dictionary with optimization results
    """
    # Initialize engine
    engine = BasketTradingEngine(prices, benchmark_returns)
    
    # Get initial weights using selected method
    if weight_method == 'mean_variance':
        initial_weights = engine.optimize_weights_mean_variance()
    elif weight_method == 'risk_parity':
        initial_weights = engine.optimize_weights_risk_parity()
    elif weight_method == 'momentum':
        initial_weights = engine.optimize_weights_momentum()
    elif weight_method == 'min_variance':
        initial_weights = engine.optimize_weights_minimum_variance()
    else:
        initial_weights = engine.optimize_weights_mean_variance()
    
    # Run optimization
    if optimization_method == 'bayesian':
        optimizer = BayesianStrategyOptimizer(
            engine=engine,
            base_weights=initial_weights,
            objective=objective
        )
        result = optimizer.optimize(n_calls=n_iterations)
        
    elif optimization_method == 'multi_objective':
        optimizer = MultiObjectiveOptimizer(
            engine=engine,
            base_weights=initial_weights,
            objectives=['sharpe', 'sortino', 'max_drawdown']
        )
        result = optimizer.optimize(n_calls=n_iterations)
    
    else:
        # Default to Bayesian
        optimizer = BayesianStrategyOptimizer(
            engine=engine,
            base_weights=initial_weights,
            objective=objective
        )
        result = optimizer.optimize(n_calls=n_iterations)
    
    # Run final backtest with optimal parameters
    # Convert numpy types in best_params
    clean_params = convert_numpy_types(result.best_params)
    
    final_strategy = TradingStrategy(
        weights=initial_weights,
        **{k: v for k, v in clean_params.items() if k != 'weights'}
    )
    final_results, final_metrics = engine.backtest_strategy(final_strategy)
    
    return {
        'optimization_result': result.to_dict(),
        'initial_weights': [float(w) for w in initial_weights],
        'final_metrics': final_metrics.to_dict(),
        'backtest_data': {
            'dates': final_results.index.strftime('%Y-%m-%d').tolist(),
            'portfolio_values': [float(v) for v in final_results['portfolio_value'].tolist()],
            'cumulative_returns': [float(v) for v in (final_results['cumulative_returns'] * 100).fillna(0).tolist()],
            'drawdown': [float(v) for v in (final_results['drawdown'] * 100).fillna(0).tolist()]
        }
    }

