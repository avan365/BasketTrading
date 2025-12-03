"""
Core basket trading engine with portfolio optimization capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_return': round(self.total_return * 100, 2),
            'annualized_return': round(self.annualized_return * 100, 2),
            'volatility': round(self.volatility * 100, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'sortino_ratio': round(self.sortino_ratio, 3),
            'max_drawdown': round(self.max_drawdown * 100, 2),
            'calmar_ratio': round(self.calmar_ratio, 3),
            'win_rate': round(self.win_rate * 100, 2),
            'profit_factor': round(self.profit_factor, 3),
            'beta': round(self.beta, 3),
            'alpha': round(self.alpha * 100, 2),
            'information_ratio': round(self.information_ratio, 3)
        }


@dataclass
class TradingStrategy:
    """Trading strategy parameters optimized via Bayesian optimization."""
    # Weight optimization
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Rebalancing parameters
    rebalance_frequency: int = 21  # Days between rebalancing
    rebalance_threshold: float = 0.05  # Min drift to trigger rebalance
    
    # Risk management
    max_position_size: float = 0.25  # Max weight for single position
    min_position_size: float = 0.02  # Min weight for single position
    stop_loss: float = 0.15  # Stop loss percentage
    take_profit: float = 0.30  # Take profit percentage
    
    # Momentum parameters
    momentum_window: int = 20  # Days for momentum calculation
    momentum_weight: float = 0.3  # How much to weight momentum in allocation
    
    # Volatility scaling
    vol_target: float = 0.15  # Target portfolio volatility
    vol_lookback: int = 21  # Days for volatility calculation
    use_vol_scaling: bool = True


class BasketTradingEngine:
    """
    Core engine for basket trading with multiple optimization strategies.
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.05
    ):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.tickers = list(prices.columns)
        self.n_assets = len(self.tickers)
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Precompute statistics
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252
        self.corr_matrix = self.returns.corr()
    
    def optimize_weights_mean_variance(
        self,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        max_weight: float = 0.25,
        min_weight: float = 0.0
    ) -> np.ndarray:
        """
        Optimize portfolio weights using mean-variance optimization.
        """
        n = self.n_assets
        
        def portfolio_volatility(weights):
            return np.sqrt(weights @ self.cov_matrix @ weights)
        
        def portfolio_return(weights):
            return weights @ self.mean_returns
        
        def neg_sharpe_ratio(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: portfolio_return(w) - target_return
            })
        
        if target_volatility is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: portfolio_volatility(w) - target_volatility
            })
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n))
        
        # Initial guess (equal weights)
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            neg_sharpe_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def optimize_weights_risk_parity(self) -> np.ndarray:
        """
        Optimize weights using risk parity (equal risk contribution).
        """
        n = self.n_assets
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            marginal_contrib = self.cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def risk_parity_objective(weights):
            rc = risk_contribution(weights)
            target_rc = 1.0 / n
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0.01, 0.5) for _ in range(n))
        x0 = np.ones(n) / n
        
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else x0
    
    def optimize_weights_momentum(
        self,
        lookback: int = 20,
        top_n: Optional[int] = None
    ) -> np.ndarray:
        """
        Optimize weights based on momentum strategy.
        """
        # Calculate momentum scores
        momentum = self.returns.iloc[-lookback:].mean()
        
        # Normalize to positive values
        momentum_shifted = momentum - momentum.min() + 0.001
        
        # If top_n specified, only keep top performers
        if top_n is not None and top_n < self.n_assets:
            threshold = momentum.nlargest(top_n).min()
            momentum_shifted[momentum < threshold] = 0
        
        # Normalize to sum to 1
        weights = momentum_shifted / momentum_shifted.sum()
        
        return weights.values
    
    def optimize_weights_minimum_variance(
        self,
        max_weight: float = 0.25
    ) -> np.ndarray:
        """
        Find minimum variance portfolio.
        """
        n = self.n_assets
        
        def portfolio_variance(weights):
            return weights @ self.cov_matrix @ weights
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, max_weight) for _ in range(n))
        x0 = np.ones(n) / n
        
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else x0
    
    def backtest_strategy(
        self,
        strategy: TradingStrategy,
        initial_capital: float = 100000
    ) -> Tuple[pd.DataFrame, PortfolioMetrics]:
        """
        Backtest a trading strategy and return performance metrics.
        """
        weights = strategy.weights
        if len(weights) != self.n_assets:
            weights = np.ones(self.n_assets) / self.n_assets
        
        # Initialize tracking
        portfolio_values = [initial_capital]
        portfolio_weights_history = [weights.copy()]
        dates = self.returns.index.tolist()
        
        current_weights = weights.copy()
        current_capital = initial_capital
        last_rebalance_idx = 0
        
        for i, date in enumerate(dates):
            daily_returns = self.returns.iloc[i].values
            
            # Calculate new portfolio value
            weighted_return = np.sum(current_weights * daily_returns)
            current_capital *= (1 + weighted_return)
            
            # Update weights based on returns (drift)
            current_weights = current_weights * (1 + daily_returns)
            current_weights = current_weights / np.sum(current_weights)
            
            # Check for rebalancing
            days_since_rebalance = i - last_rebalance_idx
            weight_drift = np.max(np.abs(current_weights - weights))
            
            should_rebalance = (
                days_since_rebalance >= strategy.rebalance_frequency or
                weight_drift >= strategy.rebalance_threshold
            )
            
            if should_rebalance and strategy.use_vol_scaling:
                # Volatility scaling
                recent_vol = self.returns.iloc[max(0, i-strategy.vol_lookback):i+1].std().mean() * np.sqrt(252)
                vol_scalar = min(2.0, max(0.5, strategy.vol_target / recent_vol)) if recent_vol > 0 else 1.0
                
                # Apply momentum adjustment
                if strategy.momentum_weight > 0 and i >= strategy.momentum_window:
                    momentum_returns = self.returns.iloc[i-strategy.momentum_window:i].mean()
                    momentum_weights = (momentum_returns - momentum_returns.min() + 0.001)
                    momentum_weights = momentum_weights / momentum_weights.sum()
                    
                    current_weights = (
                        (1 - strategy.momentum_weight) * weights +
                        strategy.momentum_weight * momentum_weights.values
                    )
                else:
                    current_weights = weights.copy()
                
                # Apply position size limits
                current_weights = np.clip(
                    current_weights,
                    strategy.min_position_size,
                    strategy.max_position_size
                )
                current_weights = current_weights / np.sum(current_weights)
                
                last_rebalance_idx = i
            
            portfolio_values.append(current_capital)
            portfolio_weights_history.append(current_weights.copy())
        
        # Create results dataframe
        results = pd.DataFrame({
            'date': [dates[0] - pd.Timedelta(days=1)] + dates,
            'portfolio_value': portfolio_values
        })
        results.set_index('date', inplace=True)
        results['returns'] = results['portfolio_value'].pct_change()
        results['cumulative_returns'] = (1 + results['returns']).cumprod() - 1
        
        # Calculate drawdown
        rolling_max = results['portfolio_value'].cummax()
        results['drawdown'] = (results['portfolio_value'] - rolling_max) / rolling_max
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        return results, metrics
    
    def _calculate_metrics(self, results: pd.DataFrame) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        returns = results['returns'].dropna()
        
        if len(returns) < 2:
            return PortfolioMetrics()
        
        # Basic metrics
        total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Drawdown metrics
        max_drawdown = abs(results['drawdown'].min())
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate and profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        total_gains = positive_returns.sum() if len(positive_returns) > 0 else 0
        total_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 1
        profit_factor = total_gains / total_losses if total_losses > 0 else 0
        
        # Benchmark-relative metrics
        beta, alpha, information_ratio = 0, 0, 0
        if self.benchmark_returns is not None:
            aligned_benchmark = self.benchmark_returns.reindex(returns.index).dropna()
            if len(aligned_benchmark) > 10:
                aligned_returns = returns.loc[aligned_benchmark.index]
                
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_var = aligned_benchmark.var()
                beta = covariance / benchmark_var if benchmark_var > 0 else 0
                
                benchmark_ann_return = aligned_benchmark.mean() * 252
                alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_ann_return - self.risk_free_rate))
                
                tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
                active_return = annualized_return - benchmark_ann_return
                information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        return PortfolioMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio
        )
    
    def get_efficient_frontier(
        self,
        n_portfolios: int = 100
    ) -> pd.DataFrame:
        """Generate efficient frontier portfolios."""
        results = []
        
        # Range of target returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        for target in target_returns:
            try:
                weights = self.optimize_weights_mean_variance(target_return=target)
                port_return = weights @ self.mean_returns
                port_vol = np.sqrt(weights @ self.cov_matrix @ weights)
                sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
                
                results.append({
                    'return': port_return,
                    'volatility': port_vol,
                    'sharpe': sharpe,
                    'weights': weights.tolist()
                })
            except Exception:
                continue
        
        return pd.DataFrame(results)

