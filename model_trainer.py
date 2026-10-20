"""
Module 4: Machine Learning Laboratory Model Training Script
Trains the predictive asset classifier on REAL historical market data instead
of synthetic/simulated feature vectors.

How this differs from the old synthetic trainer
-------------------------------------------------
Previously, every feature (RSI, MACD, P/E, news sentiment, ...) and the label
were randomly sampled from statistical distributions -- the model never saw a
real stock or a real price move.

This version:
  1. Loads the real stock universe from `data/stock_list.csv`.
  2. Downloads REAL historical OHLCV price bars for each ticker via yfinance
     (`DataFetcher.fetch_historical_ohlcv`).
  3. Downloads REAL fundamental snapshots (P/E, EPS growth, market cap,
     revenue growth, debt/equity) via yfinance (`DataFetcher.fetch_fundamental_metrics`).
  4. Computes the exact same technical indicators used at inference time
     (`IndicatorEngine`) from the real price history, walking forward through
     each ticker's timeline so every training row only uses information that
     was actually available "as of" that date (no look-ahead in the inputs).
  5. Labels every row using the REAL, realized forward return of the stock
     over the following `forward_days` trading sessions: 1 if the price
     actually went up, 0 if it did not. This is genuine supervised learning
     on real market outcomes, not a synthetic scoring formula.

News features
-------------
Point-in-time historical news archives are not available from the free
Marketaux tier, so there is no real historical news signal to attach to past
dates without look-ahead bias. Those five News_* columns are filled with the
same neutral defaults `NewsFetcher` already uses when no API key / no data is
available, keeping the feature schema identical to production. At inference
time, when `MARKETAUX_API_KEY` is set, `app.py` still fetches *real, current*
news features for live scoring -- only the *training* data lacks a real
historical news signal.

Usage
-----
    python model_trainer.py --max-tickers 60 --period 5y --forward-days 10

Requires internet access (to Yahoo Finance via yfinance). This can take a
while depending on how many tickers you train on.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from data_fetcher import DataFetcher
from indicator_engine import IndicatorEngine

FEATURE_ORDER = [
    "RSI", "MACD_Hist", "BB_Position", "SMA_Ratio", "Volume_Trend",
    "PE_Ratio", "EPS_Growth", "Market_Cap", "Revenue_Growth", "Debt_to_Equity",
    "News_Sentiment_Avg", "News_Sentiment_Weighted", "News_Count_24h",
    "News_Impact_Sum", "News_Recency_Weighted_Count",
]

# No point-in-time historical news is available (see module docstring), so
# training rows use the same neutral defaults NewsFetcher falls back to.
DEFAULT_NEWS_FEATURES = {
    "News_Sentiment_Avg": 0.0,
    "News_Sentiment_Weighted": 0.0,
    "News_Count_24h": 0.0,
    "News_Impact_Sum": 0.0,
    "News_Recency_Weighted_Count": 0.0,
}


def _label_from_forward_return(forward_return: float, up_threshold: float = 0.0) -> int:
    """1 if the real price actually rose more than `up_threshold` over the
    forward window, else 0. Using a fixed real-world threshold (0% return)
    keeps the label grounded in an actual, checkable market outcome."""
    return int(forward_return > up_threshold)


def build_ticker_training_rows(
    ticker: str,
    company_name: str,
    ohlcv_df: pd.DataFrame,
    fundamentals: dict,
    forward_days: int,
    sample_stride: int,
    min_history: int,
) -> list:
    """
    Walks forward through a single ticker's REAL historical price series and
    builds one labeled training row per sampled date.

    - Features at index i are computed only from data up to and including
      row i (no look-ahead in the inputs).
    - The label at index i is derived from the REAL close price realized
      `forward_days` trading sessions later (no look-ahead violation since
      the label, by definition, is the thing we're trying to predict).
    """
    rows = []
    if ohlcv_df.empty or len(ohlcv_df) < (min_history + forward_days + 1):
        return rows

    enriched = IndicatorEngine.compute_technical_indicators(ohlcv_df.copy())
    close = enriched["Close"].values
    n = len(enriched)

    # IndicatorEngine only produces real (non-default) indicator values once
    # there are at least 200 rows of real history, so never sample earlier
    # than that, and always stop early enough to observe a real outcome.
    start_idx = max(min_history, 200)
    end_idx = n - forward_days - 1

    for i in range(start_idx, end_idx, max(1, sample_stride)):
        window_df = enriched.iloc[: i + 1]
        features = IndicatorEngine.extract_current_features(
            window_df, fundamentals, DEFAULT_NEWS_FEATURES
        )

        entry_price = close[i]
        exit_price = close[i + forward_days]
        if entry_price == 0 or np.isnan(entry_price) or np.isnan(exit_price):
            continue

        forward_return = (exit_price - entry_price) / entry_price

        training_row = dict(features)
        training_row["Ticker"] = ticker
        training_row["Company_Name"] = company_name
        training_row["Forward_Return"] = forward_return
        training_row["Label"] = _label_from_forward_return(forward_return)
        rows.append(training_row)

    return rows


def collect_real_training_dataset(
    max_tickers: int = 60,
    history_period: str = "5y",
    forward_days: int = 10,
    sample_stride: int = 5,
    min_history: int = 200,
    fetcher: DataFetcher = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Builds a real-market training dataset by pulling genuine historical OHLCV
and fundamental data for tickers in the configured universe, engineering
    the same feature schema used at inference time, and labeling each row
    with the real subsequent price outcome.
    """
    fetcher = fetcher or DataFetcher()
    universe_df = fetcher.load_universe().head(max_tickers)

    all_rows = []
    for idx, row in universe_df.iterrows():
        ticker = row["Ticker"]
        company_name = row.get("Company", ticker)

        if verbose:
            print(f"[{idx + 1}/{len(universe_df)}] Fetching real history for {ticker}...")

        ohlcv_df = fetcher.fetch_historical_ohlcv(ticker, period=history_period)
        if ohlcv_df.empty:
            if verbose:
                print("  -> skipped (no historical data returned)")
            continue

        fundamentals = fetcher.fetch_fundamental_metrics(ticker)

        ticker_rows = build_ticker_training_rows(
            ticker=ticker,
            company_name=company_name,
            ohlcv_df=ohlcv_df,
            fundamentals=fundamentals,
            forward_days=forward_days,
            sample_stride=sample_stride,
            min_history=min_history,
        )
        if verbose:
            print(f"  -> {len(ticker_rows)} labeled samples from real price history")
        all_rows.extend(ticker_rows)

        # Be polite to the free data provider between requests.
        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError(
            "No real training data could be collected. Check your internet "
            "connection and confirm `data/stock_list.csv` contains valid, "
            "actively-traded tickers."
        )

    return pd.DataFrame(all_rows)


def build_and_serialize_model(
    max_tickers: int = 60,
    history_period: str = "5y",
    forward_days: int = 10,
    sample_stride: int = 5,
    save_dataset: bool = True,
):
    """Trains and stores the production classification engine artifact using real market history."""

    print("=" * 70)
    print("Collecting REAL historical market data via yfinance. This calls")
    print("Yahoo Finance for every ticker and can take a while.")
    print("=" * 70)

    dataset = collect_real_training_dataset(
        max_tickers=max_tickers,
        history_period=history_period,
        forward_days=forward_days,
        sample_stride=sample_stride,
    )

    if save_dataset:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        dataset_path = os.path.join(data_dir, "real_training_dataset.csv")
        dataset.to_csv(dataset_path, index=False)
        print(f"Real training dataset saved to: {dataset_path} ({len(dataset)} rows)")

    X = dataset[FEATURE_ORDER]
    y = dataset["Label"]

    print(f"Training samples: {len(X)} | Positive class ratio: {y.mean():.3f}")

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    if len(X_test) > 0 and y_test.nunique() > 1:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Hold-out accuracy on real (unseen) historical data: {acc:.3f}")
        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            print(f"Hold-out ROC AUC: {auc:.3f}")
        except Exception:
            pass

    # Refit on the full real dataset before shipping the production artifact.
    model.fit(X, y)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "stock_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model successfully saved to target destination: {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the stock model on real historical market data (via yfinance)."
    )
    parser.add_argument(
        "--max-tickers", type=int, default=60,
        help="Number of tickers from data/stock_list.csv to pull real history for (default: 60).",
    )
    parser.add_argument(
        "--period", type=str, default="5y",
        help="yfinance history period to download per ticker, e.g. 2y, 5y, max (default: 5y).",
    )
    parser.add_argument(
        "--forward-days", type=int, default=10,
        help="Number of real trading days ahead used to define the label (default: 10).",
    )
    parser.add_argument(
        "--stride", type=int, default=5,
        help="Sample every Nth trading day per ticker to reduce autocorrelation (default: 5).",
    )
    args = parser.parse_args()

    build_and_serialize_model(
        max_tickers=args.max_tickers,
        history_period=args.period,
        forward_days=args.forward_days,
        sample_stride=args.stride,
    )
