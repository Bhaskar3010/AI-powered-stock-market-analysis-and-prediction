import pytest
import pandas as pd
from data_fetcher import DataFetcher

def test_load_universe_structure():
    fetcher = DataFetcher()
    try:
        df = fetcher.load_universe()
        assert isinstance(df, pd.DataFrame)
        assert "Ticker" in df.columns
        assert "Sector" in df.columns
        assert len(df) > 0
    except FileNotFoundError:
        pytest.skip("Seed asset file database index file not found at local target paths.")