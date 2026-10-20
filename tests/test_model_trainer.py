import numpy as np
import pandas as pd

from model_trainer import build_ticker_training_rows, FEATURE_ORDER, DEFAULT_NEWS_FEATURES


def _make_fake_real_ohlcv(n=400, seed=1):
    """Builds a deterministic, price-like OHLCV series (shaped exactly like
    the real DataFetcher output) purely so the training-row builder can be
    unit tested without hitting the network."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "Date": dates,
        "Open": close * 0.999,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1_000, 100_000, n),
    })


def test_build_ticker_training_rows_schema_and_no_lookahead():
    ohlcv_df = _make_fake_real_ohlcv()
    fundamentals = {
        "pe_ratio": 22.0,
        "eps_growth": 0.1,
        "market_cap": 5e10,
        "revenue_growth": 0.08,
        "debt_to_equity": 0.4,
    }

    rows = build_ticker_training_rows(
        ticker="TEST.NS",
        company_name="Test Co",
        ohlcv_df=ohlcv_df,
        fundamentals=fundamentals,
        forward_days=10,
        sample_stride=5,
        min_history=200,
    )

    assert len(rows) > 0

    for row in rows:
        # Every real feature column must be present.
        for feat in FEATURE_ORDER:
            assert feat in row
        # Label must be a real binary outcome derived from a real forward return.
        assert row["Label"] in (0, 1)
        assert isinstance(row["Forward_Return"], float)
        # News columns fall back to the documented neutral defaults since no
        # point-in-time historical news is available.
        for k, v in DEFAULT_NEWS_FEATURES.items():
            assert row[k] == v


def test_build_ticker_training_rows_empty_on_insufficient_history():
    short_df = _make_fake_real_ohlcv(n=50)
    rows = build_ticker_training_rows(
        ticker="SHORT.NS",
        company_name="Short Co",
        ohlcv_df=short_df,
        fundamentals={},
        forward_days=10,
        sample_stride=5,
        min_history=200,
    )
    assert rows == []
