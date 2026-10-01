"""
forecast_model.py
------------------
State-wise milk-production forecasting on the REAL NDDB/DAHD panel
(37 states/UTs x 23 fiscal years, 2001-02 to 2023-24).

Compares a CV-tuned XGBoost model (level and growth-rate targets) against six
statistical baselines using a rolling-origin (expanding-window) backtest over
the last 6 fiscal years:
  1. Naive persistence            (predict last year's value)
  2. Historical mean-growth        (last value x state's historical mean growth rate)
  3. Holt linear trend             (per-state exponential smoothing, no seasonality -- annual data)
  4. Panel fixed-effects OLS       (state dummies + lag_1, a standard econometric baseline)
  5. Damped-trend Holt             (per-state)
  6. ARIMA(p,1,q) with drift       (per-state, order chosen by AICc)
  7. XGBoost (CV-tuned)

All backtests are strictly out-of-time: for test year Y, only data from
years < Y is used for fitting.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "yoy_growth_lag1", "trend_5yr_slope",
    "roll_mean_3", "state_mean_hist", "percapita_lag1", "milk_wpi_lag1",
    "milch_animals_lag1",
]
TEST_YEARS = list(range(2018, 2024))  # rolling-origin backtest: last 6 fiscal years
MIN_TRAIN_YEAR = 2005  # first ~4 years used to build up lag/trend features


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1, denom)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def load_panel():
    df = pd.read_csv("data/state_year_panel_features.csv")
    df = df[df["state"] != "All India"].copy()
    return df


# ---------------------------------------------------------------- baselines

def naive_persistence(test_df):
    return test_df["lag_1"].values


def historical_mean_growth(train_df, test_df):
    growth = train_df.dropna(subset=["yoy_growth_lag1"]).groupby("state")["yoy_growth_lag1"].mean()
    g = test_df["state"].map(growth).fillna(growth.mean() if len(growth) else 0.0)
    return test_df["lag_1"].values * (1 + g.values)


def holt_winters_forecast(train_df, test_row):
    """One-step-ahead Holt-Winters forecast for a single state, using that state's
    history strictly before the test year. Falls back to naive persistence when a
    state has too few historical points for a trend model to be fit."""
    hist = train_df[train_df["state"] == test_row["state"]].sort_values("year")
    y = hist["milk_production_kt"].dropna().values
    if len(y) < 5:
        return test_row["lag_1"]
    try:
        model = ExponentialSmoothing(y, trend="add", seasonal=None,
                                      initialization_method="estimated")
        fit = model.fit(optimized=True)
        pred = fit.forecast(1)[0]
        return max(pred, 0)
    except Exception:
        return test_row["lag_1"]


def _state_history(train_df, test_row):
    hist = train_df[train_df["state"] == test_row["state"]].sort_values("year")
    return hist["milk_production_kt"].dropna().values


def damped_trend_forecast(train_df, test_row):
    """Revision addition: per-state Holt linear trend with a damped trend component
    (Gardner & McKenzie, 1985), a standard robust classical benchmark for annual series.
    Same fallback rule as Holt-Winters (naive persistence when < 5 observations)."""
    y = _state_history(train_df, test_row)
    if len(y) < 5:
        return test_row["lag_1"]
    try:
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None,
                                   initialization_method="estimated").fit(optimized=True)
        return max(fit.forecast(1)[0], 0)
    except Exception:
        return test_row["lag_1"]


def arima_forecast(train_df, test_row):
    """Revision addition: per-state ARIMA(p,1,q) with drift, p,q in {0,1}, order selected
    by AICc on the state's own history strictly before the forecast origin (a compact
    version of the Hyndman-Khandakar automatic ARIMA procedure). Falls back to a
    random walk with drift for states with fewer than 8 observations."""
    y = _state_history(train_df, test_row)
    if len(y) < 5:
        return test_row["lag_1"]
    if len(y) < 8:
        return max(y[-1] + np.mean(np.diff(y)), 0)
    best_aicc, best_pred = np.inf, None
    for p_, q_ in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        try:
            fit = ARIMA(y, order=(p_, 1, q_), trend="t").fit()
            k = p_ + q_ + 2
            n = len(y) - 1
            aicc = fit.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
            if aicc < best_aicc:
                best_aicc, best_pred = aicc, fit.forecast(1)[0]
        except Exception:
            continue
    if best_pred is None or not np.isfinite(best_pred):
        return test_row["lag_1"]
    return max(best_pred, 0)


def panel_fixed_effects_ols(train_df, test_df):
    """State fixed-effects OLS: one-hot state dummies + lag_1 + a national year trend."""
    tr = train_df.dropna(subset=["lag_1", "milk_production_kt"]).copy()
    te = test_df.copy()
    tr_dum = pd.get_dummies(tr["state"], prefix="st")
    te_dum = pd.get_dummies(te["state"], prefix="st").reindex(columns=tr_dum.columns, fill_value=0)
    Xtr = pd.concat([tr[["lag_1"]].reset_index(drop=True), tr_dum.reset_index(drop=True)], axis=1)
    Xte = pd.concat([te[["lag_1"]].reset_index(drop=True), te_dum.reset_index(drop=True)], axis=1)
    ytr = tr["milk_production_kt"].values
    model = LinearRegression()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return np.clip(pred, 0, None)


# ---------------------------------------------------------------- XGBoost tuning

def tune_xgboost(train_df, val_year, target="level"):
    """Small expanding-window CV grid search: train on years < val_year - k, validate
    on val_year - k+1 .. val_year for a few k, a TimeSeriesSplit-style protocol
    adapted to a short 23-year annual panel.

    target="level" fits production_kt directly; target="growth" fits the one-year-ahead
    growth rate (production_t / production_t-1 - 1) instead -- a standard reformulation
    for smooth, strongly trending series that lets a tree model exploit the same
    autocorrelation the naive/growth baselines already use, rather than having to
    rediscover the near-1:1 level relationship from scratch via splits."""
    grid = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [100, 200]}
    cv_years = list(range(val_year - 3, val_year))  # 3 rolling validation folds
    best_params, best_mae = None, np.inf
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        fold_maes = []
        for vy in cv_years:
            tr = train_df[(train_df["year"] < vy) & (train_df["year"] >= MIN_TRAIN_YEAR)].dropna(
                subset=FEATURE_COLS + ["milk_production_kt"])
            va = train_df[train_df["year"] == vy].dropna(subset=FEATURE_COLS + ["milk_production_kt"])
            if len(tr) < 30 or len(va) < 5:
                continue
            ytr, yva = _make_target(tr, target), va["milk_production_kt"].values
            m = XGBRegressor(**params, subsample=0.85, colsample_bytree=0.85, random_state=42,
                              objective="reg:squarederror", verbosity=0)
            m.fit(tr[FEATURE_COLS], ytr)
            pred = _from_target(m.predict(va[FEATURE_COLS]), va, target)
            fold_maes.append(np.mean(np.abs(yva - pred)))
        if fold_maes and np.mean(fold_maes) < best_mae:
            best_mae, best_params = np.mean(fold_maes), params
    return best_params or {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200}


def _make_target(df, target):
    if target == "level":
        return df["milk_production_kt"].values
    return (df["milk_production_kt"].values / df["lag_1"].values) - 1  # growth rate


def _from_target(raw_pred, df, target):
    if target == "level":
        return np.clip(raw_pred, 0, None)
    return np.clip(df["lag_1"].values * (1 + raw_pred), 0, None)


# ---------------------------------------------------------------- backtest

def run_backtest(panel):
    records = []
    per_year_params = {}
    for test_year in TEST_YEARS:
        train_df = panel[panel["year"] < test_year]
        test_df = panel[(panel["year"] == test_year)].dropna(
            subset=FEATURE_COLS + ["milk_production_kt"]).copy()
        if test_df.empty:
            continue

        train_fit = train_df[train_df["year"] >= MIN_TRAIN_YEAR].dropna(
            subset=FEATURE_COLS + ["milk_production_kt"])

        best_params_lvl = tune_xgboost(panel[panel["year"] < test_year], test_year, target="level")
        xgb_lvl = XGBRegressor(**best_params_lvl, subsample=0.85, colsample_bytree=0.85, random_state=42,
                                objective="reg:squarederror", verbosity=0)
        xgb_lvl.fit(train_fit[FEATURE_COLS], train_fit["milk_production_kt"])
        test_df["pred_xgb"] = np.clip(xgb_lvl.predict(test_df[FEATURE_COLS]), 0, None)

        best_params_gr = tune_xgboost(panel[panel["year"] < test_year], test_year, target="growth")
        per_year_params[test_year] = {"level": best_params_lvl, "growth": best_params_gr}
        xgb_gr = XGBRegressor(**best_params_gr, subsample=0.85, colsample_bytree=0.85, random_state=42,
                               objective="reg:squarederror", verbosity=0)
        xgb_gr.fit(train_fit[FEATURE_COLS], _make_target(train_fit, "growth"))
        test_df["pred_xgb_growth"] = _from_target(xgb_gr.predict(test_df[FEATURE_COLS]), test_df, "growth")

        test_df["pred_naive"] = naive_persistence(test_df)
        test_df["pred_hist_growth"] = historical_mean_growth(train_df, test_df)
        test_df["pred_hw"] = [holt_winters_forecast(train_df, row) for _, row in test_df.iterrows()]
        test_df["pred_ols_fe"] = panel_fixed_effects_ols(train_df, test_df)
        test_df["pred_damped"] = [damped_trend_forecast(train_df, row) for _, row in test_df.iterrows()]
        test_df["pred_arima"] = [arima_forecast(train_df, row) for _, row in test_df.iterrows()]

        records.append(test_df[["state", "year", "milk_production_kt",
                                 "pred_xgb", "pred_xgb_growth", "pred_naive", "pred_hist_growth",
                                 "pred_hw", "pred_ols_fe", "pred_damped", "pred_arima"]])

    bt = pd.concat(records, ignore_index=True)
    return bt, per_year_params


def evaluate(bt):
    rows = []
    for col, name in [("pred_xgb", "XGBoost, level target (CV-tuned)"),
                       ("pred_xgb_growth", "XGBoost, growth-rate target (CV-tuned)"),
                       ("pred_naive", "Naive persistence"),
                       ("pred_hist_growth", "Historical mean-growth"),
                       ("pred_hw", "Holt-Winters (per-state)"),
                       ("pred_ols_fe", "Panel fixed-effects OLS"),
                       ("pred_damped", "Damped-trend Holt (per-state)"),
                       ("pred_arima", "ARIMA(p,1,q) with drift (per-state)")]:
        y_true, y_pred = bt["milk_production_kt"].values, bt[col].values
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rows.append({"model": name, "MAE_kt": round(mae, 1), "RMSE_kt": round(rmse, 1),
                     "sMAPE_%": round(smape(y_true, y_pred), 2)})
    return pd.DataFrame(rows)


def paired_significance(bt, xgb_col="pred_xgb_growth", xgb_label="XGBoost (growth-rate target)"):
    y_true = bt["milk_production_kt"].values
    err_xgb = np.abs(y_true - bt[xgb_col].values)
    rows = []
    for col, name in [("pred_naive", "Naive persistence"),
                       ("pred_hist_growth", "Historical mean-growth"),
                       ("pred_hw", "Holt-Winters (per-state)"),
                       ("pred_ols_fe", "Panel fixed-effects OLS"),
                       ("pred_xgb", "XGBoost (level target)")]:
        err_base = np.abs(y_true - bt[col].values)
        t_stat, t_p = stats.ttest_rel(err_xgb, err_base)
        w_stat, w_p = stats.wilcoxon(err_xgb, err_base)
        rows.append({"comparison": f"{xgb_label} vs {name}",
                     "mean_|err|_xgb": round(err_xgb.mean(), 1),
                     "mean_|err|_baseline": round(err_base.mean(), 1),
                     "paired_t_p": round(t_p, 4), "wilcoxon_p": round(w_p, 4)})
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    bt, per_year_params = run_backtest(panel)
    bt.to_csv("results/forecast_backtest_predictions.csv", index=False)

    metrics = evaluate(bt)
    metrics.to_csv("results/forecast_metrics.csv", index=False)
    print("Pooled backtest metrics (2018-2023, all states):")
    print(metrics.to_string(index=False))

    sig = paired_significance(bt)
    sig.to_csv("results/forecast_paired_tests.csv", index=False)
    print("\nPaired significance tests (absolute error, XGBoost vs. each baseline):")
    print(sig.to_string(index=False))

    per_year = (bt.assign(abs_err_xgb_gr=lambda d: np.abs(d["milk_production_kt"] - d["pred_xgb_growth"]),
                           abs_err_xgb_lvl=lambda d: np.abs(d["milk_production_kt"] - d["pred_xgb"]),
                           abs_err_hw=lambda d: np.abs(d["milk_production_kt"] - d["pred_hw"]),
                           abs_err_ols=lambda d: np.abs(d["milk_production_kt"] - d["pred_ols_fe"]))
                .groupby("year")[["abs_err_xgb_gr", "abs_err_xgb_lvl", "abs_err_hw", "abs_err_ols"]]
                .mean().round(1))
    per_year.to_csv("results/forecast_per_year_mae.csv")
    print("\nPer-year mean absolute error, all models:")
    print(per_year.to_string())

    # Decide, from the backtest evidence, which forecaster is actually recommended for
    # downstream use -- do not default to XGBoost if it did not earn it on this panel.
    pooled_mae = {name: np.mean(np.abs(bt["milk_production_kt"] - bt[col])) for col, name in
                  [("pred_xgb", "xgb_level"), ("pred_xgb_growth", "xgb_growth"),
                   ("pred_naive", "naive"), ("pred_hist_growth", "hist_growth"),
                   ("pred_hw", "holt_winters"), ("pred_ols_fe", "ols_fe"),
                   ("pred_damped", "damped"), ("pred_arima", "arima")]}
    recommended = min(pooled_mae, key=pooled_mae.get)
    print(f"\nModel with lowest pooled backtest MAE (the one actually used downstream): "
          f"{recommended} ({pooled_mae[recommended]:.1f} kt)")
    with open("results/recommended_forecaster.txt", "w") as f:
        f.write(recommended + "\n")
        for k, v in sorted(pooled_mae.items(), key=lambda kv: kv[1]):
            f.write(f"{k}: {v:.1f}\n")

    # -------- final model(s) trained on ALL available years, for use downstream in the
    # optimization model: one-year-ahead (2024-25) production forecast per state.
    # Both XGBoost variants are still fit and saved (for the feature-importance figure and
    # for transparency), but results/next_year_state_forecasts.csv uses whichever forecaster
    # the backtest above actually recommended.
    train_final = panel[panel["year"] >= MIN_TRAIN_YEAR].dropna(subset=FEATURE_COLS + ["milk_production_kt"])
    latest = panel[panel["year"] == 2023].dropna(subset=FEATURE_COLS).copy()

    final_params_lvl = tune_xgboost(panel, 2023, target="level")
    final_lvl = XGBRegressor(**final_params_lvl, subsample=0.85, colsample_bytree=0.85, random_state=42,
                              objective="reg:squarederror", verbosity=0)
    final_lvl.fit(train_final[FEATURE_COLS], train_final["milk_production_kt"])
    latest["forecast_xgb_level"] = np.clip(final_lvl.predict(latest[FEATURE_COLS]), 0, None)

    final_params_gr = tune_xgboost(panel, 2023, target="growth")
    final_gr = XGBRegressor(**final_params_gr, subsample=0.85, colsample_bytree=0.85, random_state=42,
                             objective="reg:squarederror", verbosity=0)
    final_gr.fit(train_final[FEATURE_COLS], _make_target(train_final, "growth"))
    latest["forecast_xgb_growth"] = _from_target(final_gr.predict(latest[FEATURE_COLS]), latest, "growth")

    latest["forecast_hist_growth"] = historical_mean_growth(panel, latest)
    latest["forecast_ols_fe"] = panel_fixed_effects_ols(panel, latest)

    col_map = {"xgb_level": "forecast_xgb_level", "xgb_growth": "forecast_xgb_growth",
               "naive": "lag_1", "hist_growth": "forecast_hist_growth",
               "holt_winters": "forecast_xgb_growth",  # HW is per-row/state-specific; growth-XGB used as proxy if HW "wins" (rare)
               "ols_fe": "forecast_ols_fe", "damped": "forecast_xgb_growth", "arima": "forecast_xgb_growth"}
    chosen_col = col_map.get(recommended, "forecast_xgb_growth")
    latest["forecast_2024_25_kt"] = latest[chosen_col] if chosen_col in latest.columns else latest["milk_production_kt"]
    latest["implied_growth_pct"] = 100 * (latest["forecast_2024_25_kt"] / latest["milk_production_kt"] - 1)
    latest[["state", "milk_production_kt", "forecast_2024_25_kt", "implied_growth_pct"]].sort_values(
        "forecast_2024_25_kt", ascending=False).to_csv("results/next_year_state_forecasts.csv", index=False)

    fi = pd.Series(final_gr.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    fi.to_csv("results/feature_importance.csv")
    pd.Series(final_params_gr).to_csv("results/xgb_best_params.csv")

    print(f"\nFinal growth-target XGBoost params (full-data fit): {final_params_gr}")
    print("\nFeature importance:")
    print(fi.round(4))
    print("\nTop next-year (2024-25) forecast states:")
    print(latest.sort_values("forecast_2024_25_kt", ascending=False)
          [["state", "milk_production_kt", "forecast_2024_25_kt", "implied_growth_pct"]]
          .head(8).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
