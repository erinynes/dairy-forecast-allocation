"""
sensitivity_systematic.py  (added in revision; Reviewer 1)
----------------------------------------------------------
Systematic examination of how the allocation depends on the modelling choices that
are NOT observed parameters:
  (A) budget share of full coverage (10%..50%) x per-state cap (20%..100% of budget)
  (B) alternative efficiency and equity proxies.
Outcomes are reported in real units so that they are comparable across proxies:
  - value_rs_cr      : value of the capacity installed, valued at each state's NCDFI price
                       (Rs crore/yr; capacity tonnes x 1000 L/t x Rs/L)
  - share_below_avg  : % of installed capacity placed in states whose per-capita
                       availability is below the national average (471 g/day, 2023-24)
  - states_served, Jaccard similarity of the funded-state set with the base case,
    HHI of the capacity distribution.
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

import optimize_allocation as oa

KAPPA, COST = oa.UNIT_CAPACITY_KT_YEAR, oa.UNIT_COST_RS
NAT_AVG_PC = 471.0

df = oa.build_state_inputs()
need_units = np.ceil(df["increment_kt"] / KAPPA).sum()
below = (df["percapita_gms_day"] < NAT_AVG_PC).values
price = df["avg_price_rs_per_litre"].values


def mm(v):
    v = np.asarray(v, float)
    return (v - v.min()) / (v.max() - v.min())


EFF = {
    "E1 price x increment (base)": df["efficiency_weight"].values,
    "E2 price per litre only": mm(price),
    "E3 increment volume only": mm(df["increment_kt"].values),
}
inv = 1 / df["percapita_gms_day"].values
EQ = {
    "Q1 inverse per-capita, min-max (base)": df["equity_weight"].values,
    "Q2 shortfall vs national average": np.clip(1 - df["percapita_gms_day"].values / NAT_AVG_PC, 0, None)
                                         / np.clip(1 - df["percapita_gms_day"].values / NAT_AVG_PC, 0, None).max(),
    "Q3 rank of inverse per-capita": mm(pd.Series(inv).rank().values),
}


def solve(w, budget_share, cap_share):
    n = len(df)
    budget = budget_share * need_units * COST
    ub = np.maximum(0, np.minimum(np.floor(df["increment_kt"].values / KAPPA),
                                  np.floor(cap_share * budget / COST)))
    res = milp(-w, constraints=LinearConstraint(np.full((1, n), COST), -np.inf, budget),
               bounds=Bounds(np.zeros(n), ub), integrality=np.ones(n))
    return np.round(res.x).astype(int), budget


def outcomes(x):
    cap = x * KAPPA
    tot = cap.sum()
    sh = cap / tot if tot > 0 else cap
    return {"states_served": int((x > 0).sum()),
            "capacity_kt": round(tot, 1),
            "value_rs_cr": round((cap * 1e6 * price).sum() / 1e7, 1),
            "share_below_avg_%": round(100 * cap[below].sum() / tot, 1) if tot > 0 else 0,
            "HHI": round((sh ** 2).sum(), 3),
            "funded": ";".join(df["state"][x > 0])}


def jaccard(a, b):
    a, b = set(a.split(";")) - {""}, set(b.split(";")) - {""}
    return round(len(a & b) / len(a | b), 2) if a | b else 1.0


# ---------------- (A) budget share x cap grid
rows = []
for lam in [0.0, 0.5, 1.0]:
    w = lam * df["efficiency_weight"].values + (1 - lam) * df["equity_weight"].values
    base = outcomes(solve(w, 0.25, 0.40)[0])
    for bs in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for cs in [0.20, 0.30, 0.40, 0.60, 1.00]:
            x, budget = solve(w, bs, cs)
            o = outcomes(x)
            o.update({"lambda": lam, "budget_share": bs, "cap_share": cs,
                      "budget_rs_cr": round(budget / 1e7, 1), "jaccard_vs_base": jaccard(o["funded"], base["funded"])})
            rows.append(o)
grid = pd.DataFrame(rows)
grid.to_csv("results/sensitivity_budget_cap_grid.csv", index=False)
pd.set_option("display.width", 250)
print(grid[grid["lambda"] == 0.5].drop(columns="funded").to_string(index=False))

# ---------------- (B) proxy variants
rows = []
for lam in [0.0, 0.5, 1.0]:
    base_w = lam * EFF["E1 price x increment (base)"] + (1 - lam) * EQ["Q1 inverse per-capita, min-max (base)"]
    base = outcomes(solve(base_w, 0.25, 0.40)[0])
    for en, ev in EFF.items():
        for qn, qv in EQ.items():
            if lam == 1.0 and qn[:2] != "Q1":
                continue  # equity proxy irrelevant at lambda = 1
            if lam == 0.0 and en[:2] != "E1":
                continue  # efficiency proxy irrelevant at lambda = 0
            x, _ = solve(lam * ev + (1 - lam) * qv, 0.25, 0.40)
            o = outcomes(x)
            o.update({"lambda": lam, "efficiency_proxy": en, "equity_proxy": qn,
                      "jaccard_vs_base": jaccard(o["funded"], base["funded"])})
            rows.append(o)
prox = pd.DataFrame(rows)
prox.to_csv("results/sensitivity_proxies.csv", index=False)
print(prox[["lambda", "efficiency_proxy", "equity_proxy", "states_served", "value_rs_cr",
            "share_below_avg_%", "HHI", "jaccard_vs_base", "funded"]].to_string(index=False))

# ---------------- frontier in real units (for the revised Fig.)
fr = []
for lam in np.round(np.linspace(0, 1, 11), 2):
    w = lam * df["efficiency_weight"].values + (1 - lam) * df["equity_weight"].values
    x, _ = solve(w, 0.25, 0.40)
    o = outcomes(x)
    o["lambda"] = lam
    fr.append(o)
fr = pd.DataFrame(fr)
fr.to_csv("results/allocation_frontier_real_units.csv", index=False)
print(fr.drop(columns="funded").to_string(index=False))
