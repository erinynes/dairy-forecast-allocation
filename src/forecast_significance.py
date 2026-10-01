"""
forecast_significance.py  (added in revision)
---------------------------------------------
Panel-appropriate comparison of forecast accuracy, replacing the reliance on
i.i.d. paired t / Wilcoxon tests used in the first submission (Reviewer 1, Reviewer 2).

The backtest produces a (state x test-year) panel of forecast errors. Errors of the
same state are serially correlated across test years, and errors of different states
in the same year share common shocks (e.g. COVID-19 in 2020-21), so the n = 200
state-year loss differentials are NOT independent. We therefore report, for each
pair of models (A, B), the loss differential d_it = L(e^A_it) - L(e^B_it) and:

  1. iid paired t-test and Wilcoxon signed-rank (first-submission tests, kept for reference)
  2. state-clustered t-test: regress d_it on a constant with standard errors clustered
     by state (robust to arbitrary within-state serial correlation), t with G-1 df
  3. Diebold-Mariano test on the cross-sectional average differential d_t (T = 6 test
     years), with the Harvey-Leybourne-Newbold small-sample correction; this is robust
     to cross-sectional dependence (common year shocks) but has very low power
  4. state-block bootstrap (resample whole states with their full error paths,
     B = 5000) percentile 95% CI for the mean differential
Holm's step-down correction is applied across the comparisons for test (2).

Losses: absolute error ('000 t, scale-dependent, the metric used for model selection)
and absolute scaled error (|e| divided by the state's in-sample naive MAE, i.e. the
MASE contribution of Hyndman & Koehler 2006, scale-free).
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

rng = np.random.default_rng(42)
B = 5000

MODELS = [("pred_ols_fe", "Panel fixed-effects OLS"),
          ("pred_hist_growth", "Historical mean-growth"),
          ("pred_hw", "Holt-Winters (per-state)"),
          ("pred_arima", "ARIMA(p,1,q) with drift"),
          ("pred_damped", "Damped-trend Holt"),
          ("pred_xgb_growth", "XGBoost, growth-rate target"),
          ("pred_xgb", "XGBoost, level target"),
          ("pred_naive", "Naive persistence")]

bt = pd.read_csv("results/forecast_backtest_predictions.csv")
panel = pd.read_csv("data/state_year_panel_features.csv")
panel = panel[panel["state"] != "All India"]

# MASE scale: each state's in-sample one-step naive MAE over years before the first test year
first_test = bt["year"].min()
hist = panel[panel["year"] < first_test].sort_values(["state", "year"])
scale = hist.groupby("state")["milk_production_kt"].apply(lambda s: np.mean(np.abs(np.diff(s.dropna().values))))
bt["scale"] = bt["state"].map(scale)
bt = bt[bt["scale"] > 0].copy()

y = bt["milk_production_kt"].values


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


# ---------------- metrics table incl. MASE
rows = []
for col, name in MODELS:
    e = y - bt[col].values
    rows.append({"model": name, "MAE_kt": np.mean(np.abs(e)), "RMSE_kt": np.sqrt(np.mean(e ** 2)),
                 "sMAPE_%": smape(y, bt[col].values), "MASE": np.mean(np.abs(e) / bt["scale"].values)})
met = pd.DataFrame(rows).round({"MAE_kt": 1, "RMSE_kt": 1, "sMAPE_%": 2, "MASE": 3})
met.to_csv("results/forecast_metrics_with_mase.csv", index=False)
print(met.to_string(index=False))
print(f"\nn = {len(bt)} state-year errors, {bt['state'].nunique()} states, {bt['year'].nunique()} test years")


def tests(dA, dB, df):
    d = dA - dB
    out = {"mean_diff": d.mean()}
    out["iid_t_p"] = stats.ttest_rel(dA, dB).pvalue
    out["wilcoxon_p"] = stats.wilcoxon(dA, dB).pvalue
    # state-clustered t
    X = np.ones((len(d), 1))
    fit = sm.OLS(d, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["state"])[0]},
                           use_t=True)
    G = df["state"].nunique()
    t_cl = fit.params[0] / fit.bse[0]
    out["cluster_t_p"] = 2 * stats.t.sf(abs(t_cl), G - 1)
    # DM-HLN on yearly cross-sectional means (h = 1)
    dt = pd.Series(d, index=df["year"].values).groupby(level=0).mean().values
    T = len(dt)
    dm = dt.mean() / np.sqrt(dt.var(ddof=0) / T)
    dm_hln = dm * np.sqrt((T - 1) / T)
    out["dm_hln_p"] = 2 * stats.t.sf(abs(dm_hln), T - 1)
    # state-block bootstrap
    states = df["state"].unique()
    by_state = {s: d[(df["state"] == s).values] for s in states}
    boots = np.empty(B)
    for b in range(B):
        draw = rng.choice(states, size=len(states), replace=True)
        boots[b] = np.concatenate([by_state[s] for s in draw]).mean()
    out["boot_ci_lo"], out["boot_ci_hi"] = np.percentile(boots, [2.5, 97.5])
    return out


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1, running)
    return adj


all_rows = []
for ref_col, ref_name in [("pred_ols_fe", "Panel fixed-effects OLS"),
                          ("pred_xgb_growth", "XGBoost, growth-rate target")]:
    for loss_name in ["abs_error", "abs_scaled_error"]:
        rows = []
        for col, name in MODELS:
            if col == ref_col:
                continue
            eA = np.abs(y - bt[ref_col].values)
            eB = np.abs(y - bt[col].values)
            if loss_name == "abs_scaled_error":
                eA, eB = eA / bt["scale"].values, eB / bt["scale"].values
            r = tests(eA, eB, bt)
            r.update({"reference": ref_name, "comparator": name, "loss": loss_name})
            rows.append(r)
        adj = holm([r["cluster_t_p"] for r in rows])
        for r, a in zip(rows, adj):
            r["cluster_t_p_holm"] = a
        all_rows += rows

res = pd.DataFrame(all_rows)[["reference", "comparator", "loss", "mean_diff", "iid_t_p", "wilcoxon_p",
                              "cluster_t_p", "cluster_t_p_holm", "dm_hln_p", "boot_ci_lo", "boot_ci_hi"]]
res = res.round(4)
res.to_csv("results/forecast_panel_tests.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))

# Relative (percentage) forecast errors of the selected forecaster, used to build
# the uncertainty scenarios for the stochastic allocation model (optimize_stochastic.py)
rel = bt[["state", "year", "milk_production_kt", "pred_ols_fe"]].copy()
rel["rel_error"] = rel["milk_production_kt"] / rel["pred_ols_fe"] - 1
rel.to_csv("results/ols_fe_relative_errors.csv", index=False)
print("\nOLS-FE relative errors: mean %.4f, sd %.4f, median |.| %.4f" %
      (rel["rel_error"].mean(), rel["rel_error"].std(), rel["rel_error"].abs().median()))
