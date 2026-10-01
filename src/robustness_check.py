"""
robustness_check.py
--------------------
Checks whether the "recommended forecaster" conclusion in forecast_model.py
(panel fixed-effects OLS narrowly beats growth-rate XGBoost) is an artifact
of the specific 6-year backtest window (2018-19 to 2023-24) chosen there, by
re-running the identical rolling-origin backtest over two alternative
windows: a shorter 4-year window and a longer 8-year window.
"""

import numpy as np
import pandas as pd

import forecast_model as fm

panel = fm.load_panel()

WINDOWS = {
    "4yr_2020_2023": list(range(2020, 2024)),
    "6yr_2018_2023 (main paper)": list(range(2018, 2024)),
    "8yr_2016_2023": list(range(2016, 2024)),
}

all_rows = []
for label, years in WINDOWS.items():
    fm.TEST_YEARS = years
    bt, _ = fm.run_backtest(panel)
    metrics = fm.evaluate(bt)
    metrics["window"] = label
    best_row = metrics.loc[metrics["MAE_kt"].idxmin()]
    print(f"\nWindow: {label}  (n_test_years={len(years)})")
    print(metrics.to_string(index=False))
    print(f"  -> lowest pooled MAE: {best_row['model']} ({best_row['MAE_kt']} kt)")
    all_rows.append(metrics)

out = pd.concat(all_rows, ignore_index=True)
out.to_csv("results/robustness_backtest_windows.csv", index=False)
print("\nSaved results/robustness_backtest_windows.csv")
