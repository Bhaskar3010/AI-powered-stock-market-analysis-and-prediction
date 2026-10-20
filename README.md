# 📈 AI-Driven Stock Screener (Streamlit)

A beginner-friendly stock screening web app that:

1. Loads a list of stocks
2. Fetches price data and basic fundamentals
3. Calculates technical indicators (like RSI, MACD, SMA)
4. (Optional) Fetches news sentiment features
5. Uses an ML model to rank stocks and shows the **Top 50** results

> ⚠️ **Disclaimer:** This project is for educational/informational purposes only. It is **not financial advice**.

---

## ✨ What you can do in this app

- Scan and rank stocks (Top 50) using an **AI Score (0 to 100)**
- Filter by **Sector**
- Choose a **Minimum AI Score** cutoff
- (Optional) Enable/disable **News Analysis** to control speed
- Select any stock from the results to view an **interactive price chart** (candlestick + SMA lines)

---

## 🧩 Key Features (in simple terms)

- **Technical Analysis**: RSI, MACD, Bollinger Bands, SMA using `pandas-ta`
- **ML Scoring**: Uses `models/stock_model.pkl` to generate a score
- **Trained on real market history**: the model is fit on real historical prices/fundamentals and real realized returns (see below), not simulated data
- **Fast Start**: If the ML model file is missing, the app still runs using a safe fallback scoring method
- **Toggle News**: Turning news OFF makes the app much faster

---

## 🧠 Model training: real historical data (not synthetic)

`model_trainer.py` no longer invents random feature values and formulas to
create a synthetic training set. Instead it:

1. Loads real tickers from `data/stock_list.csv`.
2. Downloads real historical price bars (`yfinance`) and real fundamental
   snapshots (P/E, EPS growth, market cap, revenue growth, debt/equity) for
   each ticker.
3. Walks forward through each ticker's real price history and computes the
   same technical indicators (`indicator_engine.py`) used at inference time,
   using only data available as of each date.
4. Labels each row with the stock's real, realized forward return over the
   next N trading days (1 = price actually rose, 0 = it didn't).
5. Trains a `RandomForestClassifier` on this real dataset and reports a
   hold-out accuracy/ROC-AUC on data the model didn't train on.

Historical, point-in-time news archives aren't available from the free
Marketaux tier, so the five `News_*` training columns use the same neutral
defaults `news_fetcher.py` falls back to when no data is available. Live
scoring in `app.py` still uses real, current news when `MARKETAUX_API_KEY`
is set — only the *training* data lacks a historical news signal.

No pre-trained `stock_model.pkl` ships in this package. Train it yourself
(requires internet access, since it calls Yahoo Finance):

```bash
pip install -r requirements.txt
python model_trainer.py --max-tickers 60 --period 5y --forward-days 10
```

This can take a while depending on `--max-tickers`. It saves the trained
model to `models/stock_model.pkl` and the assembled training rows to
`data/real_training_dataset.csv`. See `models/README.md` for all the tuning
options (`--max-tickers`, `--period`, `--forward-days`, `--stride`).

If you skip this step, the app still runs — `ai_scorer.py` automatically
falls back to a deterministic scoring formula when no model file is found.

---

## 🔧 Sidebar Toggles (very important)

### 1) Enable News Analysis
- **ON** ✅: fetches news and adds sentiment features (slower)
- **OFF** 🚀: skips news fetching (faster)

### 2) Enable Additional Filters
- **ON**: shows extra filter options (market cap category, P/E, growth ranges)
- **OFF**: only sector + AI score filtering is applied

---

## 📁 Project Structure

- `app.py` — Main Streamlit dashboard (UI + pipeline)
- `data_fetcher.py` — Loads `stock_list.csv`, fetches OHLCV and fundamentals using yfinance
- `indicator_engine.py` — Computes technical indicators and builds the feature vector
- `model_trainer.py` — Trains `stock_model.pkl` on real historical price/fundamental data (see above)
- `ai_scorer.py` — Loads the ML model and converts features into an AI score (0–100)
- `news_fetcher.py` — Fetches news sentiment features (uses an API key)
- `utils.py` — Helpers for exporting results to CSV
- `models/` — Where `stock_model.pkl` is saved after you run `model_trainer.py`
- `tests/` — Unit tests

---

## ✅ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### News API key (optional)
News analysis uses:

- `MARKETAUX_API_KEY`

If you don’t set it, news features will use safe default values.

---

## ▶️ How to run the app

If the `streamlit` command is not recognized in your terminal, run it like this:

```bash
python -m streamlit run app.py --server.port 8501
```

Open in browser:

- http://127.0.0.1:8501

---

## 🧪 Run tests (optional)

```bash
pytest -q
```

---

## 📝 Notes / Troubleshooting

- The first run can be slow because the app fetches data sequentially for multiple stocks.
- For faster results:
  - set **Enable News Analysis = OFF**
  - reduce the **Number of tickers to analyze** in the sidebar (if available)

---

## 📌 Credits

- Data: **yfinance**
- Indicators: **pandas-ta**
- UI: **Streamlit + Plotly**
- News Data: **MARKETAUX_API_KEY**
