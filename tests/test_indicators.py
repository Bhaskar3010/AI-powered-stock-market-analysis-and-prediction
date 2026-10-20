import pandas as pd
import numpy as np
from indicator_engine import IndicatorEngine


def test_technical_indicators_flat_input():
    # Construct structured mock datasets to simulate flat market conditions
    dates = pd.date_range(start="2026-01-01", periods=250, freq="D")
    mock_df = pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * 250,
        "High": [100.0] * 250,
        "Low": [100.0] * 250,
        "Close": [100.0] * 250,
        "Volume": [1000] * 250
    })

    processed_df = IndicatorEngine.compute_technical_indicators(mock_df)
    assert "RSI" in processed_df.columns
    assert "SMA_50" in processed_df.columns
    assert "SMA_200" in processed_df.columns
    assert float(processed_df['RSI'].iloc[-1]) == 50.0


def test_extract_current_features_includes_news_defaults():
    dates = pd.date_range(start="2026-01-01", periods=250, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * 250,
        "High": [100.0] * 250,
        "Low": [100.0] * 250,
        "Close": [100.0] * 250,
        "Volume": [1000] * 250
    })
    df = IndicatorEngine.compute_technical_indicators(df)

    fundamentals = {
        "pe_ratio": 25.0,
        "eps_growth": 0.05,
        "market_cap": 1e10,
        "revenue_growth": 0.05,
        "debt_to_equity": 0.5,
    }

    feats = IndicatorEngine.extract_current_features(df, fundamentals)
    # News fields should exist with defaults
    assert "News_Sentiment_Avg" in feats
    assert "News_Count_24h" in feats

