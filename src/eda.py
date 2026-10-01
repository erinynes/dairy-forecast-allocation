"""
eda.py
------
Descriptive statistics and regional-disparity analysis of the real
state-year milk production panel: national growth trend, top-state
concentration, regional dispersion (coefficient of variation, Gini
coefficient of state shares), and correlation between production growth
and livestock-population / price covariates.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)


def gini(shares):
    x = np.sort(np.asarray(shares, dtype=float))
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def main():
    panel = pd.read_csv("data/state_year_panel_raw.csv")
    national = pd.read_csv("data/national_milk_production.csv")

    # 1. National CAGR over full window vs a normal reference (2001-02 -> 2023-24)
    start, end = national.iloc[national["year"] == "2001-02"], national.iloc[national["year"] == "2023-24"]
    cagr = (end["production_million_tonnes"].values[0] / start["production_million_tonnes"].values[0]) ** (1 / 22) - 1
    print(f"National milk production CAGR, 2001-02 to 2023-24: {cagr*100:.2f}% per year")

    # 2. Top-5 state concentration over time
    conc = (panel[panel["state"] != "All India"]
            .groupby("year").apply(
                lambda d: d.nlargest(5, "milk_production_kt")["milk_production_kt"].sum()
                / d["milk_production_kt"].sum() * 100, include_groups=False))
    conc.name = "top5_share_pct"
    print("\nTop-5 state share of national production, selected years:")
    print(conc.loc[[2001, 2010, 2023]].round(1))

    # 3. Regional dispersion: CV and Gini of state production shares, by year
    def year_stats(d):
        shares = d["milk_production_kt"] / d["milk_production_kt"].sum()
        return pd.Series({
            "cv_of_state_shares": shares.std() / shares.mean(),
            "gini_of_state_shares": gini(shares.values),
        })
    disp = (panel[panel["state"] != "All India"].dropna(subset=["milk_production_kt"])
            .groupby("year").apply(year_stats, include_groups=False))
    print("\nRegional dispersion (state shares of national output), selected years:")
    print(disp.loc[[2001, 2010, 2023]].round(3))

    # 4. Growth heterogeneity: state-level CAGR, fastest vs slowest.
    # Restricted to states/UTs with >=20 years of coverage so short, administratively
    # truncated series (Ladakh split off 2019, Daman & Diu merged away 2020, etc.) don't
    # produce spurious "fastest growing" artifacts from a tiny base / short window.
    def state_cagr(d):
        d = d.dropna(subset=["milk_production_kt"]).sort_values("year")
        if len(d) < 20 or d["milk_production_kt"].iloc[0] <= 0:
            return np.nan
        n_yrs = d["year"].iloc[-1] - d["year"].iloc[0]
        if n_yrs <= 0:
            return np.nan
        return (d["milk_production_kt"].iloc[-1] / d["milk_production_kt"].iloc[0]) ** (1 / n_yrs) - 1

    cagrs = (panel[panel["state"] != "All India"].groupby("state").apply(state_cagr, include_groups=False)
             .dropna().sort_values(ascending=False))
    cagrs.name = "cagr"
    print("\nFastest-growing states/UTs with >=20 yrs coverage (CAGR, full window):")
    print((cagrs.head(8) * 100).round(2))
    print("\nSlowest-growing / declining states/UTs with >=20 yrs coverage:")
    print((cagrs.tail(8) * 100).round(2))

    # 5. Correlation: national production YoY growth vs national milk WPI YoY change
    # (does a price index rise coincide with output growth, or the reverse -- output growth
    # softening prices? both are plausible a priori for this market).
    wpi = pd.read_csv("data/milk_wholesale_price_index.csv")
    milk_wpi_row = wpi[wpi["particulars"] == "Milk"].iloc[0]
    wpi_years = [c for c in wpi.columns if c not in ("particulars", "weight")]
    wpi_series = pd.Series([float(milk_wpi_row[c]) for c in wpi_years],
                            index=[int(c.split("-")[0]) for c in wpi_years]).sort_index()
    wpi_yoy = wpi_series.pct_change()

    nat = national.copy()
    nat["year"] = nat["year"].apply(lambda y: int(y.split("-")[0]))
    nat = nat.set_index("year").sort_index()
    growth_nat = nat["production_million_tonnes"].pct_change()

    merged = pd.concat([wpi_yoy.rename("wpi_yoy"), growth_nat.rename("prod_yoy")], axis=1).dropna()
    corr = merged["wpi_yoy"].corr(merged["prod_yoy"])
    print(f"\nCorrelation (national milk WPI YoY change vs national production YoY growth, "
          f"{merged.index.min()}-{merged.index.max()}): r = {corr:.3f}  (n={len(merged)})")

    # Save summary tables for the paper
    conc.to_csv("results/top5_concentration.csv")
    disp.to_csv("results/regional_dispersion.csv")
    cagrs.to_csv("results/state_cagr.csv")
    merged.to_csv("results/wpi_vs_growth.csv")
    print("\nSaved summary tables to results/")


if __name__ == "__main__":
    main()
