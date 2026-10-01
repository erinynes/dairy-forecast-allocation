import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 300})


def year_to_int(y):
    return int(y.split("-")[0])


def fig0_pipeline_overview():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    stage_color = "#1f77b4"
    stage_face = "#eaf2fb"
    input_color = "#888888"
    input_face = "#f4f4f4"
    text_color = "#1a1a1a"

    def stage_box(x, y, w, h, title, lines, face=stage_face, edge=stage_color, title_color=stage_color):
        box = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top", fontsize=9.5,
                 fontweight="bold", color=title_color, zorder=3)
        ax.text(x + w / 2, y + h - 7.2, "\n".join(lines), ha="center", va="top", fontsize=7.6,
                 color=text_color, zorder=3, linespacing=1.5)

    def arrow(x0, y0, x1, y1, color="#333333"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6,
                                      shrinkA=0, shrinkB=0, mutation_scale=14), zorder=2)

    # Main left-to-right pipeline stages
    stage_box(2, 24, 20, 18, "1. Data",
               ["Real NDDB/DAHD panel:", "37 states/UTs x 23 yrs.", "Production, per-capita", "availability, livestock,", "prices (Sec. 3.1)"])
    stage_box(27, 24, 22, 18, "2. Forecasting",
               ["XGBoost (2 targets) vs.", "6 statistical baselines;", "rolling-origin backtest +", "panel tests select the", "forecaster (Sec. 3.2)"])
    stage_box(54, 24, 22, 18, "3. Optimization",
               ["Deterministic MILP and", "two-stage stochastic", "MILP (backtest-error", "scenarios) (Sec. 3.3)"])
    stage_box(81, 24, 17, 18, "4. Outputs",
               ["Efficiency-equity", "frontier, VSS/EVPI,", "illustrative integer", "allocations (Sec. 4)"], face="#fdf1e3", edge="#c9791f", title_color="#c9791f")

    arrow(22, 33, 27, 33)
    arrow(49, 33, 54, 33)
    arrow(76, 33, 81, 33)

    # label on the arrow between stages 2 and 3
    ax.text(51.5, 31.2, "point forecast\n+ error\nscenarios", ha="center", va="top", fontsize=6.8,
             color="#333333", style="italic")

    # Side inputs feeding the optimization stage
    stage_box(48, 2, 32, 14, "Real-data inputs to optimization",
               ["NCDFI consumer prices -> efficiency weight", "Per-capita milk availability -> equity weight",
                "NABARD BMC unit cost & capacity; budget, cap"],
               face=input_face, edge=input_color, title_color="#555555")
    arrow(64, 16, 64, 24, color=input_color)

    fig.tight_layout()
    fig.savefig("figures/fig0_pipeline_overview.png")
    plt.close(fig)


def fig1_national_trend_and_disparity():
    nat = pd.read_csv("data/national_milk_production.csv")
    nat["yr"] = nat["year"].apply(year_to_int)
    disp = pd.read_csv("results/regional_dispersion.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(nat["yr"], nat["production_million_tonnes"], color="#1f77b4", linewidth=1.8)
    axes[0].set_title("India: national milk production, 1991-92 to 2023-24")
    axes[0].set_ylabel("Million tonnes")
    axes[0].set_xlabel("Fiscal year (starting)")

    axes[1].plot(disp["year"], disp["gini_of_state_shares"], marker="o", color="#d62728",
                 markersize=3, linewidth=1.4)
    axes[1].set_title("Regional concentration of production\n(Gini coefficient of state shares)")
    axes[1].set_xlabel("Fiscal year (starting)")
    axes[1].set_ylabel("Gini coefficient")
    fig.tight_layout()
    fig.savefig("figures/fig1_national_trend_and_disparity.png")
    plt.close(fig)


def fig2_forecast_model_comparison():
    m = pd.read_csv("results/forecast_metrics_with_mase.csv").sort_values("MAE_kt")
    m["model"] = m["model"].str.replace("Holt-Winters (per-state)", "Holt linear trend", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, col, lab in [(axes[0], "MAE_kt", "MAE ('000 tonnes)"), (axes[1], "MASE", "MASE (scale-free)")]:
        cols = ["#1f4e79" if "OLS" in n else ("#4a90d9" if "XGBoost" in n else "#9a9a9a") for n in m["model"]]
        ax.barh(m["model"], m[col], color=cols)
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        for i, v in enumerate(m[col]):
            ax.text(v, i, f" {v:.0f}" if col == "MAE_kt" else f" {v:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, m[col].max() * 1.18)
    axes[1].set_yticklabels([])
    axes[0].set_title("Pooled MAE, test years 2018-19 to 2023-24", fontsize=9.5)
    axes[1].set_title("Pooled MASE (scale-free)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig("figures/fig2_forecast_model_comparison.png", dpi=300)
    plt.close(fig)


def _old_fig2():
    m = pd.read_csv("results/forecast_metrics.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#1f77b4", "#4a90d9", "#888888", "#d62728", "#2ca02c", "#9467bd"]
    ax.bar(m["model"], m["MAE_kt"], color=colors[:len(m)])
    ax.set_ylabel("Mean absolute error ('000 tonnes)")
    ax.set_title("Rolling-origin backtest (2018-2023): forecast accuracy by model")
    ax.tick_params(axis="x", rotation=30, labelsize=7.5)
    for i, v in enumerate(m["MAE_kt"]):
        ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig("figures/fig2_forecast_model_comparison.png")
    plt.close(fig)


def fig3_per_year_mae():
    py = pd.read_csv("results/forecast_per_year_mae.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(py["year"], py["abs_err_xgb_gr"], marker="o", label="XGBoost (growth-rate target)",
            color="#1f77b4")
    ax.plot(py["year"], py["abs_err_ols"], marker="s", label="Panel fixed-effects OLS",
            color="#d62728")
    ax.plot(py["year"], py["abs_err_hw"], marker="^", label="Holt linear trend (per-state)",
            color="#2ca02c")
    ax.axvspan(2019.5, 2020.5, color="grey", alpha=0.15)
    ax.annotate("COVID-19\ndisruption year", xy=(2020, ax.get_ylim()[1]*0.9), fontsize=7.5,
                ha="center", color="dimgray")
    ax.set_xlabel("Test (fiscal) year")
    ax.set_ylabel("Mean absolute error ('000 tonnes)")
    ax.set_title("Forecast error by year: all models degrade in the 2020 shock year")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig3_per_year_mae.png")
    plt.close(fig)


def fig4_feature_importance():
    fi = pd.read_csv("results/feature_importance.csv", index_col=0).iloc[:, 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(fi.index, fi.values, color="#1f77b4")
    ax.set_xlabel("Importance (gain-based)")
    ax.set_title("XGBoost (growth-rate target) feature importance")
    fig.tight_layout()
    fig.savefig("figures/fig4_feature_importance.png")
    plt.close(fig)


def fig5_frontier_and_sensitivity():
    fr = pd.read_csv("results/allocation_frontier_real_units.csv")
    grid = pd.read_csv("results/sensitivity_budget_cap_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    sc = ax.scatter(fr["share_below_avg_%"], fr["value_rs_cr"], c=fr["lambda"], cmap="viridis", s=45, zorder=3)
    ax.plot(fr["share_below_avg_%"], fr["value_rs_cr"], color="grey", lw=0.8, zorder=2)
    for _, r in fr.iterrows():
        if r["lambda"] in (0.0, 0.5, 1.0):
            off = {0.0: (-95, 4), 0.5: (8, -12), 1.0: (6, -14)}[r["lambda"]]
            ax.annotate(f"λ={r['lambda']:.1f} ({int(r['states_served'])} states)", (r["share_below_avg_%"], r["value_rs_cr"]),
                        textcoords="offset points", xytext=off, fontsize=7.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("λ")
    ax.set_xlabel("Capacity placed in below-average-availability states (%)")
    ax.set_ylabel("Value of installed capacity (Rs crore/yr)")
    ax.set_title("(a) Frontier in physical/monetary units", fontsize=9.5)
    ax = axes[1]
    g = grid[grid["lambda"] == 0.5].pivot(index="cap_share", columns="budget_share", values="states_served")
    sh = grid[grid["lambda"] == 0.5].pivot(index="cap_share", columns="budget_share", values="share_below_avg_%")
    im = ax.imshow(g.values, cmap="Blues", aspect="auto", origin="lower", vmin=0)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            ax.text(j, i, f"{int(v)}\n{sh.values[i, j]:.0f}%", ha="center", va="center", fontsize=6.8,
                    color="white" if v > 7 else "black")
    ax.set_xticks(range(g.shape[1]), [f"{int(100*c)}%" for c in g.columns])
    ax.set_yticks(range(g.shape[0]), [f"{int(100*c)}%" for c in g.index])
    ax.set_xlabel("Budget as share of full-coverage cost")
    ax.set_ylabel("Per-state cap (share of budget)")
    ax.set_title("(b) λ = 0.5: states funded / % below-average", fontsize=9.5)
    ax.add_patch(plt.Rectangle((2.5, 1.5), 1, 1, fill=False, ec="#d62728", lw=1.8))
    fig.tight_layout()
    fig.savefig("figures/fig5_frontier_sensitivity.png", dpi=300)
    plt.close(fig)


def fig5_allocation_frontier():
    fr = pd.read_csv("results/allocation_frontier.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sc = ax.scatter(fr["equity_score"], fr["efficiency_score"], c=fr["lambda"], cmap="viridis",
                     s=60, zorder=3)
    ax.plot(fr["equity_score"], fr["efficiency_score"], color="grey", linewidth=0.8, zorder=2)
    for _, r in fr.iterrows():
        if r["lambda"] in (0.0, 0.5, 1.0):
            ax.annotate(f"λ={r['lambda']:.1f}", (r["equity_score"], r["efficiency_score"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("λ (1 = pure efficiency, 0 = pure equity)")
    ax.set_xlabel("Equity score (per-capita-availability-weighted captured tonnage)")
    ax.set_ylabel("Efficiency score (price-weighted captured tonnage)")
    ax.set_title("Chilling-capacity investment: efficiency-equity trade-off frontier")
    fig.tight_layout()
    fig.savefig("figures/fig5_allocation_frontier.png")
    plt.close(fig)


def fig6_state_cagr():
    cagr = pd.read_csv("results/state_cagr.csv")
    cagr.columns = ["state", "cagr"]
    cagr["cagr_pct"] = cagr["cagr"] * 100
    top = pd.concat([cagr.nlargest(6, "cagr_pct"), cagr.nsmallest(6, "cagr_pct")])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in top["cagr_pct"]]
    ax.barh(top["state"], top["cagr_pct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("CAGR, 2001-02 to 2023-24 (%/year)")
    ax.set_title("Fastest- and slowest-growing states/UTs (≥20 years' coverage)")
    fig.tight_layout()
    fig.savefig("figures/fig6_state_cagr.png")
    plt.close(fig)


def fig7_district_case_study():
    mh = pd.read_csv("results/maharashtra_district_detail.csv")
    top = mh.nlargest(10, "y2024").sort_values("y2024")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].barh(top["District"], top["y2024"], color="#1f77b4")
    axes[0].set_xlabel("2023-24 milk production ('000 tonnes)")
    axes[0].set_title("Maharashtra: top-10 districts by\nmilk production, 2023-24")

    summ = pd.read_csv("results/district_case_study_summary.csv")
    colors = ["#1f77b4", "#4a90d9", "#888888"]
    axes[1].bar(summ["state"], summ["gini_latest"], color=colors[:len(summ)])
    axes[1].axhline(0.674, color="#d62728", linestyle="--", linewidth=1.2,
                     label="National state-level Gini, 2023-24 (0.674)")
    axes[1].set_ylabel("District-level Gini coefficient")
    axes[1].set_title("Within-state (district-level) concentration\nvs. between-state concentration")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig("figures/fig7_district_case_study.png")
    plt.close(fig)


if __name__ == "__main__":
    fig0_pipeline_overview()
    fig1_national_trend_and_disparity()
    fig2_forecast_model_comparison()
    fig3_per_year_mae()
    fig4_feature_importance()
    fig5_allocation_frontier()
    fig6_state_cagr()
    fig7_district_case_study()
    fig5_frontier_and_sensitivity()
    print("Saved figures to figures/")
