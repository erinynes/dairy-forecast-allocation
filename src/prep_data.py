"""
prep_data.py
------------
Loads the real government dairy datasets, reshapes the state-wise milk
production table into a tidy (state, year) panel, merges in per-capita
availability, livestock-population (census-year, forward/back-filled --
census is quinquennial not annual, this is an explicit, disclosed
simplification), and the national milk price index, then engineers the
lag/trend features used by the forecasting model in forecast_model.py.

All inputs are the real NDDB/DAHD tables saved in data/ (see data/SOURCES.md).
"""

import numpy as np
import pandas as pd

STATE_COL = "State"
EXCLUDE_ROWS = {"All India"}  # keep the national row separate; states/UTs only in the panel


def year_to_int(y):
    """Convert 'YYYY-YY' fiscal-year label to the starting calendar year as int."""
    return int(y.split("-")[0])


def load_state_panel():
    wide = pd.read_csv("data/state_milk_production.csv")
    year_cols = [c for c in wide.columns if c != STATE_COL]
    long = wide.melt(id_vars=STATE_COL, value_vars=year_cols,
                      var_name="fy", value_name="milk_production_kt")
    long = long[~long[STATE_COL].isin(EXCLUDE_ROWS)].copy()
    long["year"] = long["fy"].apply(year_to_int)
    long = long.rename(columns={STATE_COL: "state"})
    long["milk_production_kt"] = pd.to_numeric(long["milk_production_kt"], errors="coerce")
    return long[["state", "year", "fy", "milk_production_kt"]].sort_values(["state", "year"])


def load_percapita_panel():
    wide = pd.read_csv("data/state_percapita_availability.csv")
    year_cols = [c for c in wide.columns if c != STATE_COL]
    long = wide.melt(id_vars=STATE_COL, value_vars=year_cols,
                      var_name="fy", value_name="percapita_gms_day")
    long = long[~long[STATE_COL].isin(EXCLUDE_ROWS)].copy()
    long["year"] = long["fy"].apply(year_to_int)
    long = long.rename(columns={STATE_COL: "state"})
    long["percapita_gms_day"] = pd.to_numeric(long["percapita_gms_day"], errors="coerce")
    return long[["state", "year", "percapita_gms_day"]]


def load_national_price_index():
    df = pd.read_csv("data/milk_wholesale_price_index.csv")
    milk_row = df[df["particulars"] == "Milk"].iloc[0]
    year_cols = [c for c in df.columns if c not in ("particulars", "weight")]
    out = pd.DataFrame({
        "fy": year_cols,
        "milk_wpi": [float(milk_row[c]) for c in year_cols],
    })
    out["year"] = out["fy"].apply(year_to_int)
    return out[["year", "milk_wpi"]]


def load_livestock_census():
    df = pd.read_csv("data/livestock_population_census.csv")
    row = df[df["species"] == "Adult_Female_Cattle"].iloc[0]
    buf = df[df["species"] == "Adult_Female_Buffalo"].iloc[0]
    census_years = [int(c) for c in df.columns if c != "species"]
    combined = pd.DataFrame({
        "year": census_years,
        "adult_female_cattle_m": [float(row[str(y)]) for y in census_years],
        "adult_female_buffalo_m": [float(buf[str(y)]) for y in census_years],
    })
    combined["milch_animals_m"] = combined["adult_female_cattle_m"] + combined["adult_female_buffalo_m"]
    return combined


def national_milch_animals_by_year(years):
    """Interpolate the quinquennial national milch-animal count onto an annual index.
    This is an explicit, disclosed simplification (see data/SOURCES.md): the Livestock
    Census is not annual, so we linearly interpolate between census years and hold the
    last known value flat outside the observed census range."""
    census = load_livestock_census().sort_values("year")
    s = pd.Series(census["milch_animals_m"].values, index=census["year"].values)
    full_index = pd.Index(sorted(set(list(s.index) + list(years))))
    s = s.reindex(full_index).interpolate(method="index").ffill().bfill()
    return s.reindex(years)


def build_panel():
    milk = load_state_panel()
    pc = load_percapita_panel()
    panel = milk.merge(pc, on=["state", "year"], how="left")

    price = load_national_price_index()
    panel = panel.merge(price, on="year", how="left")
    # WPI series only starts 2012-13; back-fill with earliest available index value scaled
    # is NOT done -- we leave NaN and drop rows lacking price coverage for models that need it,
    # documented in the paper as the reason the price-aware model has a shorter training window.

    years = sorted(panel["year"].unique())
    milch = national_milch_animals_by_year(years)
    panel["national_milch_animals_m"] = panel["year"].map(milch)

    # National totals for reference / share-of-national feature
    national = milk[milk["state"] != "All India"].groupby("year")["milk_production_kt"].sum()
    panel["state_share_of_national_pct"] = panel.apply(
        lambda r: 100 * r["milk_production_kt"] / national.get(r["year"], np.nan)
        if pd.notna(r["milk_production_kt"]) else np.nan, axis=1)

    panel = panel.sort_values(["state", "year"]).reset_index(drop=True)
    return panel


def engineer_features(panel):
    df = panel.copy()
    df = df.sort_values(["state", "year"])
    g = df.groupby("state")["milk_production_kt"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_3"] = g.shift(3)
    df["yoy_growth_lag1"] = g.shift(1) / g.shift(2) - 1
    df["trend_5yr_slope"] = g.transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).apply(
            lambda w: np.polyfit(np.arange(len(w)), w, 1)[0] if w.notna().sum() >= 3 else np.nan,
            raw=False))
    df["roll_mean_3"] = g.shift(1).rolling(3, min_periods=2).mean()
    df["state_mean_hist"] = g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # Lag-1 versions of the exogenous/national covariates -- using the SAME-year value of
    # per-capita availability or the price index as a predictor of the same year's production
    # would leak information (per-capita availability is arithmetically derived from
    # production; price and production are set simultaneously in-year), so every exogenous
    # feature used by the forecasting model is lagged by one year.
    df["percapita_lag1"] = g_pc = df.groupby("state")["percapita_gms_day"].shift(1)
    price_by_year = df.groupby("year")["milk_wpi"].first().sort_index()
    df["milk_wpi_lag1"] = df["year"].map(price_by_year.shift(1))
    milch_by_year = df.groupby("year")["national_milch_animals_m"].first().sort_index()
    df["milch_animals_lag1"] = df["year"].map(milch_by_year.shift(1))
    return df


def main():
    panel = build_panel()
    panel.to_csv("data/state_year_panel_raw.csv", index=False)

    featured = engineer_features(panel)
    featured.to_csv("data/state_year_panel_features.csv", index=False)

    print(f"Panel: {panel['state'].nunique()} states/UTs x "
          f"{panel['year'].nunique()} years ({panel['year'].min()}-{panel['year'].max()+1} FY), "
          f"{len(panel)} state-year observations "
          f"({panel['milk_production_kt'].notna().sum()} non-missing production values).")
    print("\nSample rows:")
    print(featured[featured["state"] == "Gujarat"].tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
