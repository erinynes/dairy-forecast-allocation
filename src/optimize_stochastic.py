"""
optimize_stochastic.py  (added in revision; Reviewer 2 (b), Reviewer 3)
-----------------------------------------------------------------------
Two-stage stochastic-programming version of the allocation model, so that forecast
errors enter the optimization instead of being ignored.

First stage  : x_s  in Z+   BMC units built in state s (here-and-now, before the
                            production increment is observed)
Second stage : u_sw >= 0    capacity actually utilised in state s under scenario w
                            ('000 t/yr), u_sw <= kappa * x_s,  u_sw <= Delta_sw

    max  sum_w p_w sum_s w_s(lambda) * u_sw / kappa
    s.t. sum_s c * x_s <= B ;  x_s <= floor(0.40 B / c) ;  linking constraints above

Scenarios: realised production of state s is F_s (1 + e_sw), where F_s is the
panel-FE point forecast and e_sw is a relative forecast error of the SAME model taken
from the out-of-time backtest (results/ols_fe_relative_errors.csv). The realised
increment is Delta_sw = max(0, F_s (1 + e_sw) - P_s).
  * "historical year-block" scenarios (main): scenario w = backtest year t, every
    state takes its own error from year t (6 equiprobable scenarios). This keeps the
    state-specific error scale AND the cross-state correlation of errors (e.g. the
    common 2020-21 COVID-19 shock).
  * "independent bootstrap" scenarios (robustness): 1000 scenarios, each state's error
    drawn independently with replacement from its own six backtest errors.

The deterministic model of Section 3.3 (x_s <= floor(Delta_hat_s / kappa)) is the
expected-value (EV) problem. We report RP (stochastic solution value), EEV (expected
value of the EV solution), WS (wait-and-see), VSS = RP - EEV and EVPI = WS - RP
(Birge & Louveaux, 2011), plus expected capacity utilisation.
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack, eye

import optimize_allocation as oa

KAPPA = oa.UNIT_CAPACITY_KT_YEAR
COST = oa.UNIT_COST_RS
rng = np.random.default_rng(2026)


def inputs():
    df = oa.build_state_inputs()
    total_need_units = np.ceil(df["increment_kt"] / KAPPA).sum()
    budget = oa.BUDGET_SHARE_OF_FULL_COVERAGE * total_need_units * COST
    errs = pd.read_csv("results/ols_fe_relative_errors.csv")
    E = errs.pivot(index="state", columns="year", values="rel_error").reindex(df["state"])
    assert not E.isna().any().any(), "every candidate state needs 6 backtest errors"
    return df, budget, E


def scenario_increments(df, E, kind):
    F = df["forecast_2024_25_kt"].values if "forecast_2024_25_kt" in df else None
    P = df["milk_production_kt"].values
    F = P + df["increment_kt"].values  # point forecast (increment>0 for all candidates)
    if kind == "year_block":
        errs = E.values.T                     # S x n
    else:
        idx = rng.integers(0, E.shape[1], size=(1000, E.shape[0]))
        errs = E.values[np.arange(E.shape[0])[None, :], idx]
    inc = np.maximum(0.0, F[None, :] * (1 + errs) - P[None, :])
    return inc                                 # S x n ('000 t)


def weights(df, lam):
    return lam * df["efficiency_weight"].values + (1 - lam) * df["equity_weight"].values


def ub_cap(budget, n):
    return np.floor(oa.EQUITY_CAP_SHARE * budget / COST) * np.ones(n)


def solve_deterministic(df, lam, budget, inc):
    n = len(df)
    ub = np.maximum(0, np.minimum(np.floor(inc / KAPPA), ub_cap(budget, n)))
    res = milp(-weights(df, lam), constraints=LinearConstraint(np.full((1, n), COST), -np.inf, budget),
               bounds=Bounds(np.zeros(n), ub), integrality=np.ones(n))
    return np.round(res.x).astype(int)


def solve_sp(df, lam, budget, INC):
    S, n = INC.shape
    p = 1.0 / S
    w = weights(df, lam)
    nv = n + S * n
    c = np.concatenate([np.zeros(n), -np.tile(w / KAPPA, S) * p])
    # u_sw - kappa x_s <= 0
    A_link = hstack([-KAPPA * csr_matrix(np.tile(np.eye(n), (S, 1))), eye(S * n, format="csr")])
    A_bud = hstack([csr_matrix(np.full((1, n), COST)), csr_matrix((1, S * n))])
    A = vstack([A_link, A_bud]).tocsr()
    lb_c = np.concatenate([np.full(S * n, -np.inf), [-np.inf]])
    ub_c = np.concatenate([np.zeros(S * n), [budget]])
    lb = np.zeros(nv)
    ub = np.concatenate([ub_cap(budget, n), INC.reshape(-1)])
    integ = np.concatenate([np.ones(n), np.zeros(S * n)])
    res = milp(c, constraints=LinearConstraint(A, lb_c, ub_c), bounds=Bounds(lb, ub),
               integrality=integ, options={"time_limit": 600, "mip_rel_gap": 1e-6})
    assert res.status in (0, 1), res.message
    return np.round(res.x[:n]).astype(int)


def evaluate(df, lam, x, INC):
    w = weights(df, lam)
    util = np.minimum(KAPPA * x[None, :], INC)          # S x n
    obj = (util / KAPPA * w[None, :]).sum(axis=1).mean()
    installed = KAPPA * x.sum()
    exp_util = util.sum(axis=1).mean()
    return obj, exp_util, installed


def main():
    df, budget, E = inputs()
    print(f"Budget Rs {budget/1e7:.1f} crore, {len(df)} candidate states")
    point_inc = df["increment_kt"].values
    rows, alloc_rows = [], []
    for kind in ["year_block", "bootstrap"]:
        INC = scenario_increments(df, E, kind)
        if kind == "year_block":
            sc = pd.DataFrame(INC, columns=df["state"], index=sorted(E.columns))
            sc.round(1).to_csv("results/stochastic_scenario_increments.csv")
        for lam in [0.0, 0.5, 1.0]:
            x_ev = solve_deterministic(df, lam, budget, point_inc)
            x_sp = solve_sp(df, lam, budget, INC)
            eev, util_ev, inst_ev = evaluate(df, lam, x_ev, INC)
            rp, util_sp, inst_sp = evaluate(df, lam, x_sp, INC)
            ws = np.mean([evaluate(df, lam, solve_deterministic(df, lam, budget, INC[k]), INC[k:k + 1])[0]
                          for k in range(INC.shape[0])])
            rows.append({"scenarios": kind, "S": INC.shape[0], "lambda": lam,
                         "EEV": eev, "RP": rp, "WS": ws, "VSS": rp - eev,
                         "VSS_%_of_EEV": 100 * (rp - eev) / eev if eev > 0 else np.nan,
                         "EVPI": ws - rp,
                         "util_%_EV": 100 * util_ev / inst_ev, "util_%_SP": 100 * util_sp / inst_sp,
                         "exp_utilised_kt_EV": util_ev, "exp_utilised_kt_SP": util_sp,
                         "units_EV": int(x_ev.sum()), "units_SP": int(x_sp.sum()),
                         "spend_cr_EV": x_ev.sum() * COST / 1e7, "spend_cr_SP": x_sp.sum() * COST / 1e7,
                         "states_EV": int((x_ev > 0).sum()), "states_SP": int((x_sp > 0).sum())})
            for s, a, b in zip(df["state"], x_ev, x_sp):
                alloc_rows.append({"scenarios": kind, "lambda": lam, "state": s, "units_EV": a, "units_SP": b})
            print(rows[-1])
    out = pd.DataFrame(rows).round(3)
    out.to_csv("results/stochastic_vss.csv", index=False)
    al = pd.DataFrame(alloc_rows)
    al.to_csv("results/stochastic_allocations.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    print(al[(al["scenarios"] == "year_block") & ((al.units_EV > 0) | (al.units_SP > 0))]
          .pivot(index="state", columns="lambda", values=["units_EV", "units_SP"]).fillna(0).astype(int))


if __name__ == "__main__":
    main()
