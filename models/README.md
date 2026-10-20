# models/

This folder is where `model_trainer.py` saves `stock_model.pkl`.

No pre-trained `stock_model.pkl` ships in this package anymore. The previous
file was fit on synthetic/simulated data, which defeats the purpose of a
"real historical data" model, so it has been removed rather than replaced
with another fake artifact.

## To generate a real, data-trained model

From the project root, with internet access (the trainer calls Yahoo Finance
via `yfinance`):

```bash
pip install -r requirements.txt
python model_trainer.py --max-tickers 60 --period 5y --forward-days 10
```

This downloads real historical price bars and fundamentals for tickers in
`data/stock_list.csv`, engineers the same features used at inference time,
labels each row with the real forward return that actually occurred, trains
a `RandomForestClassifier` on it, prints a hold-out accuracy/ROC-AUC, and
saves the result here as `stock_model.pkl`. It also writes the assembled
training rows to `data/real_training_dataset.csv` for inspection.

## If you skip this step

`app.py` and `ai_scorer.py` both handle a missing model file gracefully: the
app still runs, just using the deterministic fallback scoring formula
instead of the ML model, and will tell you the ML score isn't active.

## Tuning knobs

- `--max-tickers`: how many tickers (from the top of `stock_list.csv`) to
  pull real history for. Higher = more training data, but slower and more
  Yahoo Finance requests. Start small (e.g. 30-60) to sanity check before
  running against the full universe.
- `--period`: how much real history per ticker (`2y`, `5y`, `max`, ...).
- `--forward-days`: how many real trading days ahead defines "did the stock
  go up." Shorter = more responsive/noisy; longer = smoother/slower signal.
- `--stride`: samples every Nth trading day per ticker so consecutive rows
  aren't near-duplicates of each other.
