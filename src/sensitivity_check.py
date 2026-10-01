"""
sensitivity_check.py
---------------------
Tests whether the balanced (lambda=0.5) allocation recommendation in
optimize_allocation.py is sensitive to the two real-world unit-economics
assumptions carried over from the NABARD model bankable project (unit cost
Rs. 11.78 lakh, unit capacity 1,600 L/day) and to the illustrative budget
size, by perturbing each +/-20% (unit cost, unit capacity) or +/-10 pp
(budget share) one at a time and re-solving the balanced MILP allocation.
"""

import numpy as np
import pandas as pd

import optimize_allocation as oa

df = oa.build_state_inputs()
val_per_tonne = oa.value_per_tonne_rs()

base_cost = oa.UNIT_COST_RS
base_cap_lpd = oa.UNIT_CAPACITY_LPD
base_budget_share = oa.BUDGET_SHARE_OF_FULL_COVERAGE


def total_need_rs(unit_cost, unit_cap_kt):
    total_need_units = np.ceil(df["increment_kt"] / unit_cap_kt).sum()
    return total_need_units * unit_cost


scenarios = [
    ("Base case", base_cost, base_cap_lpd, base_budget_share),
    ("Unit cost +20%", base_cost * 1.2, base_cap_lpd, base_budget_share),
    ("Unit cost -20%", base_cost * 0.8, base_cap_lpd, base_budget_share),
    ("Unit capacity +20%", base_cost, base_cap_lpd * 1.2, base_budget_share),
    ("Unit capacity -20%", base_cost, base_cap_lpd * 0.8, base_budget_share),
    ("Budget share +10pp (35%)", base_cost, base_cap_lpd, base_budget_share + 0.10),
    ("Budget share -10pp (15%)", base_cost, base_cap_lpd, base_budget_share - 0.10),
]

rows = []
for label, unit_cost, unit_cap_lpd, budget_share in scenarios:
    unit_cap_kt = unit_cap_lpd * 365 / 1_000_000
    oa.UNIT_COST_RS = unit_cost
    oa.UNIT_CAPACITY_KT_YEAR = unit_cap_kt

    budget_rs = budget_share * total_need_rs(unit_cost, unit_cap_kt)
    x_bal, ok = oa.solve_milp(df, 0.5, budget_rs, val_per_tonne)
    n_states = int((x_bal > 0).sum())
    captured_kt = (x_bal * unit_cap_kt).sum()
    top_state_idx = np.argsort(-x_bal)[:3]
    top_states = ", ".join(f"{df.iloc[i]['state']} ({x_bal[i]})" for i in top_state_idx if x_bal[i] > 0)

    rows.append({"scenario": label, "budget_rs_cr": round(budget_rs / 1e7, 1),
                 "states_served": n_states, "captured_kt": round(captured_kt, 1),
                 "top_3_states_units": top_states, "solved_ok": ok})

# restore module globals
oa.UNIT_COST_RS = base_cost
oa.UNIT_CAPACITY_KT_YEAR = base_cap_lpd * 365 / 1_000_000

out = pd.DataFrame(rows)
out.to_csv("results/allocation_sensitivity.csv", index=False)
print(out.to_string(index=False))
print("\nSaved results/allocation_sensitivity.csv")
