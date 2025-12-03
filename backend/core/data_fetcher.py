"""
Data fetching module for retrieving historical stock data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor


class DataFetcher:
    """Fetches and preprocesses financial data for basket trading."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def fetch_single_stock(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical data for a single stock."""
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                return pd.DataFrame()
            
            # Clean and standardize column names
            df.columns = df.columns.str.lower()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            self._cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_basket_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for a basket of stocks."""
        basket_data = {}
        
        for ticker in tickers:
            df = self.fetch_single_stock(ticker, start_date, end_date)
            if not df.empty:
                basket_data[ticker] = df
        
        return basket_data
    
    def get_aligned_prices(
        self,
        basket_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Get aligned closing prices for all stocks in basket."""
        prices = pd.DataFrame()
        
        for ticker, df in basket_data.items():
            if not df.empty and 'close' in df.columns:
                prices[ticker] = df['close']
        
        # Forward fill missing values, then drop any remaining NaN
        prices = prices.ffill().dropna()
        return prices
    
    def get_aligned_returns(
        self,
        basket_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Get aligned returns for all stocks in basket."""
        returns = pd.DataFrame()
        
        for ticker, df in basket_data.items():
            if not df.empty and 'returns' in df.columns:
                returns[ticker] = df['returns']
        
        returns = returns.dropna()
        return returns
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        if len(df) < 20:
            return df
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def get_benchmark_data(
        self,
        benchmark: str = "SPY",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Fetch benchmark data for comparison."""
        return self.fetch_single_stock(benchmark, start_date, end_date)


# Default popular stock baskets
DEFAULT_BASKETS = {
    "tech_leaders": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP"],
    "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO"],
    "consumer": ["WMT", "PG", "KO", "PEP", "COST", "HD", "MCD"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "diversified": ["AAPL", "JPM", "JNJ", "XOM", "WMT", "GOOGL", "PG", "UNH"]
}

