"""
optimize_allocation.py
-----------------------
The "automation" / decision-support layer: an optimization model that
consumes the forecasting pipeline's output (results/next_year_state_forecasts.csv,
the state-wise incremental milk production the backtest-selected forecaster
projects for next year) and recommends how to allocate a constrained national
capital budget for new Bulk Milk Cooler (BMC) village-level chilling
infrastructure across states, so that forecasted new production is actually
collected/chilled rather than spoiling before reaching a processing plant.

Real, cited unit economics (not fabricated):
  - BMC unit capacity and project cost are taken from NABARD's "Model Bankable
    Project on Bulk Milk Cooling Units" (agritech.tnau.ac.in/banking/nabard_pdf/
    Animal husbandry/7.Model_scheme_on_Bulk_Milk_Cooling_Centers.pdf): a 2000
    litres/day (LPD) BMC unit costs Rs. 11.78 lakh and is sized for a village/
    cooperative procuring roughly 1400-1800 L/day (we use the midpoint, 1600 L/day).
  - Milk density is approximated as 1 L = 1 kg (standard industry approximation;
    actual raw milk density is ~1.03 kg/L, a <3% simplification we disclose).
  - The illustrative national capital budget for the scenario is sized as a
    fraction of the total cost that WOULD be required to fully chill-cover every
    state's ML-forecasted production increment (so the allocation problem is
    genuinely budget-constrained, not slack) -- this scenario-sizing approach,
    not a specific appropriated government budget figure, is stated explicitly
    in the paper as a modeling choice, consistent with NPDD's real per-district
    central-assistance ceiling of Rs. 15 crore (dahd.gov.in NPDD Operational
    Guidelines) being of a comparable order of magnitude.

Decision-support design: rather than a single efficiency-only allocation, the
model sweeps a lambda in [0,1] trading off (a) region-differentiated economic
value captured, using REAL state-level consumer milk prices published by
NCDFI (data/ncdfi_state_milk_prices.csv, National Cooperative Dairy
Federation of India, civil consumer price list), against (b) equity-weighted
nutritional impact, weighting each captured tonne by the inverse of the
state's REAL per-capita milk availability (so a captured tonne in a
low-availability state like Bihar or an NE state counts for more than one in
an already milk-abundant state like Punjab or Haryana). Two states with
forecasted growth (Arunachal Pradesh, Puducherry) are not covered by the
NCDFI price list; for these the disclosed fallback is the national average
price across all NCDFI-listed federations (Rs 62.66/litre). The sweep
produces an efficiency-equity trade-off frontier.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# ---- real, cited unit economics ----
UNIT_CAPACITY_LPD = 1600.0          # litres/day, midpoint of NABARD's 1400-1800 L/day BMC catchment
UNIT_CAPACITY_KT_YEAR = UNIT_CAPACITY_LPD * 365 / 1_000_000  # '000 tonnes/year (1 L ~= 1 kg)
UNIT_COST_RS = 11.78e5              # Rs. per unit (NABARD model BMC project, 2000 LPD design)
BUDGET_SHARE_OF_FULL_COVERAGE = 0.25  # illustrative scenario: budget covers 25% of total need
EQUITY_CAP_SHARE = 0.40             # no single state may receive more than 40% of the budget

LAMBDA_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)  # 0 = pure equity, 1 = pure economic efficiency


def value_per_tonne_rs():
    """Implied national average producer value per tonne of milk, derived from REAL
    NDDB/MOSPI data: value of milk-group output (Rs crore) / national production (tonnes),
    latest overlapping year (2022-23)."""
    out_val = pd.read_csv("data/livestock_output_value.csv")
    row = out_val[out_val["year"] == "2022-23"].iloc[0]
    milk_value_rs = row["milk_group"] * 1e7  # Rs crore -> Rs
    nat_prod = pd.read_csv("data/national_milk_production.csv")
    prod_tonnes = nat_prod[nat_prod["year"] == "2022-23"]["production_million_tonnes"].values[0] * 1e6
    return milk_value_rs / prod_tonnes


def build_state_inputs():
    fc = pd.read_csv("results/next_year_state_forecasts.csv")
    fc["increment_kt"] = (fc["forecast_2024_25_kt"] - fc["milk_production_kt"]).clip(lower=0)

    pc = pd.read_csv("data/state_percapita_availability.csv")[["State", "2023-24"]].rename(
        columns={"State": "state", "2023-24": "percapita_gms_day"})
    df = fc.merge(pc, on="state", how="left")
    df = df.dropna(subset=["percapita_gms_day"])
    df = df[df["increment_kt"] > 0].reset_index(drop=True)  # only states with forecasted growth need new capacity

    # Real state-level consumer milk prices (NCDFI civil consumer price list). Two
    # forecasted-growth states (Arunachal Pradesh, Puducherry) are not covered by this
    # source; for those we disclose and fall back to the national average price computed
    # across all NCDFI-listed federation/variant price observations.
    price = pd.read_csv("data/state_avg_milk_price.csv")[["state", "avg_price_rs_per_litre"]]
    ncdfi_raw = pd.read_csv("data/ncdfi_state_milk_prices.csv")
    national_avg_price = round(ncdfi_raw["Price_Rs"].mean(), 2)
    df = df.merge(price, on="state", how="left")
    df["price_is_national_fallback"] = df["avg_price_rs_per_litre"].isna()
    df["avg_price_rs_per_litre"] = df["avg_price_rs_per_litre"].fillna(national_avg_price)

    # Equity weight: inverse of real per-capita milk availability, min-max normalized to [0,1]
    # so a below-median-availability state (e.g. Bihar, Manipur) scores high and an
    # already milk-abundant state (Punjab, Haryana) scores low.
    inv_pc = 1.0 / df["percapita_gms_day"]
    df["equity_weight"] = (inv_pc - inv_pc.min()) / (inv_pc.max() - inv_pc.min())

    # Efficiency weight: real region-differentiated economic value of the forecasted
    # increment (increment tonnes, approximating 1 L ~= 1 kg, valued at that state's REAL
    # NCDFI consumer milk price), min-max normalized to [0,1]. This replaces a
    # price-blind "share of increment" proxy used in an earlier iteration of this model,
    # so that "efficient" spend chases states where the ML-forecasted new supply is both
    # large AND commands a higher regional price, rather than volume alone.
    econ_value_rs = df["increment_kt"] * 1e6 * df["avg_price_rs_per_litre"]  # kt -> L (1L~=1kg), x Rs/L
    df["efficiency_weight"] = (econ_value_rs - econ_value_rs.min()) / (econ_value_rs.max() - econ_value_rs.min())
    return df


def solve_lp(df, lam, budget_rs, val_per_tonne):
    """Continuous relaxation: x_s = number of BMC units (fractional) built in state s."""
    n = len(df)
    max_units_by_increment = (df["increment_kt"] / UNIT_CAPACITY_KT_YEAR).values
    max_units_by_equity = (EQUITY_CAP_SHARE * budget_rs / UNIT_COST_RS) * np.ones(n)
    ub = np.minimum(max_units_by_increment, max_units_by_equity)

    obj_coef = lam * df["efficiency_weight"].values + (1 - lam) * df["equity_weight"].values
    c = -obj_coef  # linprog minimizes, we want to maximize

    A_ub = [np.full(n, UNIT_COST_RS)]  # budget constraint
    b_ub = [budget_rs]
    bounds = [(0, u) for u in ub]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    x = np.clip(res.x, 0, None) if res.success else np.zeros(n)
    return x


def solve_milp(df, lam, budget_rs, val_per_tonne):
    """Integer version: BMC units must be whole numbers -- the deployable recommendation."""
    n = len(df)
    max_units_by_increment = np.floor(df["increment_kt"] / UNIT_CAPACITY_KT_YEAR).values
    max_units_by_equity = np.floor((EQUITY_CAP_SHARE * budget_rs / UNIT_COST_RS) * np.ones(n))
    ub = np.minimum(max_units_by_increment, max_units_by_equity)
    ub = np.maximum(ub, 0)

    obj_coef = lam * df["efficiency_weight"].values + (1 - lam) * df["equity_weight"].values
    c = -obj_coef

    constraints = LinearConstraint(A=np.full((1, n), UNIT_COST_RS), lb=-np.inf, ub=budget_rs)
    bounds = Bounds(lb=np.zeros(n), ub=ub)
    integrality = np.ones(n)

    res = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
    x = np.round(res.x).astype(int) if res.success else np.zeros(n, dtype=int)
    return x, res.success


def summarize(df, x, budget_rs, val_per_tonne, label):
    captured_kt = (x * UNIT_CAPACITY_KT_YEAR).sum()
    spend_rs = (x * UNIT_COST_RS).sum()
    n_states_served = (x > 1e-6).sum()
    equity_captured = (x * UNIT_CAPACITY_KT_YEAR * df["equity_weight"].values).sum()
    efficiency_captured = (x * UNIT_CAPACITY_KT_YEAR * df["efficiency_weight"].values).sum()
    econ_value_rs = captured_kt * 1e3 * val_per_tonne
    return {"scenario": label, "budget_utilization_%": round(100 * spend_rs / budget_rs, 1),
            "states_served": int(n_states_served), "captured_kt": round(captured_kt, 1),
            "economic_value_rs_cr": round(econ_value_rs / 1e7, 2),
            "equity_score": round(equity_captured, 1),
            "efficiency_score": round(efficiency_captured, 1)}


def main():
    df = build_state_inputs()
    val_per_tonne = value_per_tonne_rs()
    print(f"Implied national average milk value: Rs {val_per_tonne:,.0f} per tonne "
          f"(2022-23, NDDB/MOSPI output-value / national production)")
    print(f"BMC unit: {UNIT_CAPACITY_LPD:.0f} L/day capacity "
          f"({UNIT_CAPACITY_KT_YEAR*1000:.1f} tonnes/year), Rs {UNIT_COST_RS/1e5:.2f} lakh/unit "
          f"(NABARD model bankable project, 2000 LPD design)")
    n_fallback = int(df["price_is_national_fallback"].sum())
    print(f"\nStates with forecasted positive growth needing new chilling capacity: {len(df)} "
          f"({n_fallback} priced via national-average fallback, no NCDFI listing)")
    print(df[["state", "milk_production_kt", "increment_kt", "percapita_gms_day",
              "avg_price_rs_per_litre", "equity_weight", "efficiency_weight"]]
          .round(2).to_string(index=False))

    total_need_units = np.ceil(df["increment_kt"] / UNIT_CAPACITY_KT_YEAR).sum()
    total_need_rs = total_need_units * UNIT_COST_RS
    budget_rs = BUDGET_SHARE_OF_FULL_COVERAGE * total_need_rs
    print(f"\nFull national coverage of all forecasted increments would need "
          f"~{total_need_units:.0f} BMC units (Rs {total_need_rs/1e7:.1f} crore).")
    print(f"Illustrative scenario budget ({BUDGET_SHARE_OF_FULL_COVERAGE:.0%} of full coverage): "
          f"Rs {budget_rs/1e7:.1f} crore")

    # ---- efficiency-equity Pareto frontier (continuous LP relaxation, fast sweep) ----
    frontier_rows = []
    for lam in LAMBDA_GRID:
        x = solve_lp(df, lam, budget_rs, val_per_tonne)
        row = summarize(df, x, budget_rs, val_per_tonne, f"lambda={lam}")
        row["lambda"] = lam
        frontier_rows.append(row)
    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv("results/allocation_frontier.csv", index=False)
    print("\nEfficiency-equity Pareto frontier (LP relaxation):")
    print(frontier[["lambda", "captured_kt", "efficiency_score", "equity_score",
                     "states_served"]].to_string(index=False))

    # ---- two representative integer (deployable) solutions: pure-efficiency vs balanced ----
    x_eff, ok1 = solve_milp(df, 1.0, budget_rs, val_per_tonne)
    x_bal, ok2 = solve_milp(df, 0.5, budget_rs, val_per_tonne)
    x_eq, ok3 = solve_milp(df, 0.0, budget_rs, val_per_tonne)

    alloc = df[["state", "increment_kt", "percapita_gms_day"]].copy()
    alloc["units_pure_efficiency"] = x_eff
    alloc["units_balanced"] = x_bal
    alloc["units_pure_equity"] = x_eq
    alloc = alloc[(alloc["units_pure_efficiency"] > 0) | (alloc["units_balanced"] > 0) |
                  (alloc["units_pure_equity"] > 0)].sort_values("units_balanced", ascending=False)
    alloc.to_csv("results/allocation_recommendations.csv", index=False)
    print(f"\nInteger (deployable) allocations solved: efficiency={ok1}, balanced={ok2}, equity={ok3}")
    print("\nBalanced (lambda=0.5) integer allocation, states receiving >=1 unit:")
    print(alloc.to_string(index=False))

    for label, x in [("Pure efficiency (lambda=1.0)", x_eff),
                      ("Balanced (lambda=0.5)", x_bal),
                      ("Pure equity (lambda=0.0)", x_eq)]:
        s = summarize(df, x, budget_rs, val_per_tonne, label)
        print(f"\n{label}: {s}")


if __name__ == "__main__":
    main()
