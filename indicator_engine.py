"""
Module 2 & 3: Financial Engineering & Feature Compilation Matrix
Utilizes pandas_ta and pandas vectors to transform raw metrics into structured machine learning features.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


class IndicatorEngine:
    @staticmethod
    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates and appends technical analytical data to the core tracking dataframe."""
        if df.empty or len(df) < 200:
            # Inject structural default vectors to guarantee continuity during edge-case initialization failures
            df['RSI'] = 50.0
            df['MACD'] = 0.0
            df['MACD_Signal'] = 0.0
            df['MACD_Hist'] = 0.0
            df['BB_Upper'] = df['Close'] * 1.05 if not df.empty else 1.0
            df['BB_Lower'] = df['Close'] * 0.95 if not df.empty else 1.0
            df['BB_Middle'] = df['Close'] if not df.empty else 1.0
            df['SMA_50'] = df['Close'] if not df.empty else 1.0
            df['SMA_200'] = df['Close'] if not df.empty else 1.0
            return df

        # Vectorized calculations via pandas_ta
        df['RSI'] = ta.rsi(df['Close'], length=14).fillna(50.0)

        macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            df['MACD'] = macd_df.iloc[:, 0].fillna(0.0)
            df['MACD_Hist'] = macd_df.iloc[:, 1].fillna(0.0)
            df['MACD_Signal'] = macd_df.iloc[:, 2].fillna(0.0)
        else:
            df['MACD'] = df['MACD_Hist'] = df['MACD_Signal'] = 0.0

        bb_df = ta.bbands(df['Close'], length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            df['BB_Lower'] = bb_df.iloc[:, 0].fillna(df['Close'])
            df['BB_Middle'] = bb_df.iloc[:, 1].fillna(df['Close'])
            df['BB_Upper'] = bb_df.iloc[:, 2].fillna(df['Close'])
        else:
            df['BB_Lower'] = df['BB_Middle'] = df['BB_Upper'] = df['Close']

        df['SMA_50'] = ta.sma(df['Close'], length=50).fillna(df['Close'])
        df['SMA_200'] = ta.sma(df['Close'], length=200).fillna(df['Close'])

        return df

    @staticmethod
    def extract_current_features(
        df: pd.DataFrame,
        fundamental_metrics: dict,
        news_features: dict | None = None,
    ) -> dict:
        """Compiles the mathematical feature vector using the last available historical row data."""

        news_features = news_features or {}

        # Defaults if DF is empty
        if df.empty:
            return {
                "RSI": 50.0,
                "MACD_Hist": 0.0,
                "BB_Position": 0.5,
                "SMA_Ratio": 1.0,
                "Volume_Trend": 1.0,
                "PE_Ratio": 25.0,
                "EPS_Growth": 0.05,
                "Market_Cap": 1e10,
                "Revenue_Growth": 0.05,
                "Debt_to_Equity": 0.5,
                # News features
                "News_Sentiment_Avg": float(news_features.get("News_Sentiment_Avg", 0.0)),
                "News_Sentiment_Weighted": float(news_features.get("News_Sentiment_Weighted", 0.0)),
                "News_Count_24h": float(news_features.get("News_Count_24h", 0.0)),
                "News_Impact_Sum": float(news_features.get("News_Impact_Sum", 0.0)),
                "News_Recency_Weighted_Count": float(news_features.get("News_Recency_Weighted_Count", 0.0)),
            }

        last_row = df.iloc[-1]

        # Prevent division by zero errors during evaluation
        bb_denom = (last_row['BB_Upper'] - last_row['BB_Lower'])
        bb_position = (last_row['Close'] - last_row['BB_Lower']) / bb_denom if bb_denom != 0 else 0.5

        sma_ratio = last_row['SMA_50'] / last_row['SMA_200'] if last_row['SMA_200'] != 0 else 1.0

        # Volume relative trend assessment
        recent_vol = df['Volume'].tail(5).mean()
        baseline_vol = df['Volume'].tail(20).mean()
        vol_trend = recent_vol / baseline_vol if baseline_vol != 0 else 1.0

        return {
            "RSI": float(last_row['RSI']),
            "MACD_Hist": float(last_row['MACD_Hist']),
            "BB_Position": float(bb_position),
            "SMA_Ratio": float(sma_ratio),
            "Volume_Trend": float(vol_trend),
            "PE_Ratio": float(fundamental_metrics.get("pe_ratio", 25.0)),
            "EPS_Growth": float(fundamental_metrics.get("eps_growth", 0.05)),
            "Market_Cap": float(fundamental_metrics.get("market_cap", 1e10)),
            "Revenue_Growth": float(fundamental_metrics.get("revenue_growth", 0.05)),
            "Debt_to_Equity": float(fundamental_metrics.get("debt_to_equity", 0.5)),
            # News features
            "News_Sentiment_Avg": float(news_features.get("News_Sentiment_Avg", 0.0)),
            "News_Sentiment_Weighted": float(news_features.get("News_Sentiment_Weighted", 0.0)),
            "News_Count_24h": float(news_features.get("News_Count_24h", 0.0)),
            "News_Impact_Sum": float(news_features.get("News_Impact_Sum", 0.0)),
            "News_Recency_Weighted_Count": float(news_features.get("News_Recency_Weighted_Count", 0.0)),
        }

