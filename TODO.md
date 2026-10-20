# TODO

## Rewrite code for 500-stock universe (stock_list.csv now has 500 rows)

- [x] Confirmed `data/stock_list.csv` has 500 rows with columns: `Ticker`, `Company`, `Sector`.
- [x] `app.py`: change `row['Company Name']` -> `row['Company']`.
- [x] `app.py`: change sidebar slider `max_value` from 1000 -> 500.
- [x] `model_trainer.py`: change `row.get("Company Name", ticker)` -> `row.get("Company", ticker)`.
- [x] `launch_streamlit.py`: update `PROJECT_DIR` from `project_1000_News_Filter_real_data\project_1000_News_Filter` -> `project_NSE_500_News_Filter_real_data\project_NSE_500_News_Filter`.
- [x] `start_streamlit.bat`: update hardcoded `project_1000_News_Filter` paths -> `project_NSE_500_News_Filter` paths.
- [x] Verify no remaining stale references and run a sanity check.

## Previously completed work

- [x] Switched model training from synthetic to real historical data (see git history / README).
