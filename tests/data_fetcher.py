"""
Module 1: High-Performance Data Fetcher Engine
Responsible for loading the stock universe and fetching clean data frames from the yfinance API.
"""
import os
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any

class DataFetcher:
    def __init__(self, stock_list_path: str = None):
        if stock_list_path is None:
            self.stock_list_path = os.path.join(os.path.dirname(__file__), 'data', 'stock_list.csv')
        else:
            self.stock_list_path = stock_list_path

    def load_universe(self) -> pd.DataFrame:
        """Loads and returns the predefined configuration universe dataframe."""
        if not os.path.exists(self.stock_list_path):
            raise FileNotFoundError(f"Universe seed file not located at: {self.stock_list_path}")
        return pd.read_csv(self.stock_list_path)

    def fetch_historical_ohlcv(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetches historical daily bar data for a single ticker instrument.
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d")
            if df.empty:
                return pd.DataFrame()
            # Standardize structural labels
            df.reset_index(inplace=True)
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols if col != 'Date'):
                return pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()

    def fetch_fundamental_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieves foundational accounting ratios using the yfinance backend engine.
        """
        default_metrics = {
            "pe_ratio": 25.0,
            "eps_growth": 0.05,
            "market_cap": 1e10,
            "revenue_growth": 0.05,
            "debt_to_equity": 0.5
        }
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or not isinstance(info, dict):
                return default_metrics
            
            return {
                "pe_ratio": info.get("trailingPE", default_metrics["pe_ratio"]),
                "eps_growth": info.get("earningsGrowth", default_metrics["eps_growth"]),
                "market_cap": info.get("marketCap", default_metrics["market_cap"]),
                "revenue_growth": info.get("revenueGrowth", default_metrics["revenue_growth"]),
                "debt_to_equity": info.get("debtToEquity", default_metrics["debt_to_equity"] * 100) / 100.0
            }
        except Exception:
            return default_metrics