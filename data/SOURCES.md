# Data sources (all real, publicly published Government of India statistics)

All figures below were transcribed directly from official statistics published by the
National Dairy Development Board (NDDB, a statutory body under the Ministry of Fisheries,
Animal Husbandry and Dairying) and the Department of Animal Husbandry and Dairying (DAHD).
No values are synthetic, interpolated, or estimated by us unless explicitly noted.

1. **state_milk_production.csv** — State/UT-wise estimates of milk production ('000 tonnes),
   2001-02 to 2023-24. Source: NDDB Statistics portal, https://www.nddb.coop/information/stats/milkprodstate,
   original data from Basic Animal Husbandry Statistics (BAHS), MoFAHD/DAHD, Government of India.

2. **state_percapita_availability.csv** — Per-capita milk availability by state/UT (grams/day),
   2009-10 to 2023-24. Source: https://www.nddb.coop/information/stats/percapitavail,
   Basic Animal Husbandry Statistics, MoFAHD/DAHD.

3. **national_milk_production.csv** — All-India annual milk production and per-capita
   availability, 1991-92 to 2023-24. Source: https://www.nddb.coop/information/stats/milkprodindia.

4. **livestock_population_census.csv** — Livestock population by species from the
   quinquennial Livestock Census, 1951 to 2019 (most recent published census; the 21st
   census was under way as of 2024-25 and not yet published at time of writing).
   Source: https://www.nddb.coop/information/stats/pop, Livestock Census, MoFAHD/DAHD.

5. **livestock_output_value.csv** — Value of output from the livestock sector by category
   (Rs. crore, current prices), 2011-12 to 2022-23. Source:
   https://www.nddb.coop/information/stats/outputvalue, National Accounts Statistics, MOSPI.

6. **milk_wholesale_price_index.csv** — Index numbers of wholesale prices (base varies by
   series) for milk and related commodity groups, 2012-13 to 2023-24. Source:
   https://www.nddb.coop/information/stats/indexwholesale, Office of the Economic Adviser,
   Government of India.

7. **ncdfi_state_milk_prices.csv** — State/federation-wise retail (civil consumer) milk
   prices (Rs./litre) by cooperative federation and fat/SNF variant, ~100 observations
   across 27 states/UTs. Source: National Cooperative Dairy Federation of India (NCDFI),
   Civil Consumer Prices, https://www.ncdfi.coop/index.php/civil-consumer-prices/.
   These are retail consumer prices, not farm-gate/procurement prices. Two forecasted-
   growth states used in the optimization model (Arunachal Pradesh, Puducherry) are not
   covered by this source; **state_avg_milk_price.csv** gives the per-state average price
   (mean across listed federations/variants) computed from this file, and the optimization
   code (optimize_allocation.py) discloses and applies a national-average-price fallback
   (Rs. 62.66/litre, the mean across all raw NCDFI observations) for the two uncovered
   states rather than fabricating a state-specific figure.

## District-level case study (data/district_case_study/, used in Section 4.6)

8. **maharashtra_district_milk_production.csv** — District-wise ESTIMATED MILK PRODUCTION
   ('000 M.T.), 35 districts, 2017-18 to 2023-24. Actual production data, not a proxy.
   Source: Maharashtra Animal Husbandry, Dairy Development and Fisheries Department, via
   the Open Government Data (OGD) Platform India,
   https://www.data.gov.in/resource/maharashtra-district-wise-estimated-milk-production-2017-18-2023-24.
   The 2023-24 district sum (16,045.4 '000 t) matches the independently sourced state-level
   figure in state_milk_production.csv (16,045.0 '000 t) to within 0.00%, a useful internal
   cross-check.

9. **punjab_district_cows.csv** — District-wise COW population (excludes buffalo), 22
   districts, Livestock Census years 1972-2019 (only the 2019 column is used). This is a
   milk-production PROXY, not a production series, and understates milk-relevant livestock
   in Punjab specifically because buffalo supply the majority of the state's milk; used and
   disclosed as such. Source: Punjab Open Government Data Platform,
   https://punjab.data.gov.in/catalog/district-wise-livestock-and-poultry-statistics-punjab.

10. **andhra_pradesh_district_livestock_2019.csv** — District-wise cattle, buffalo, sheep,
    goat, and other livestock/poultry population, 13 (pre-2022, undivided) districts, 20th
    Livestock Census (2019). Cattle+buffalo population is used as a milk-production PROXY,
    disclosed as such (not a production series). Source: Open Government Data (OGD)
    Platform India,
    https://www.data.gov.in/resource/district-wise-details-livestock-population-andhra-pradesh-20th-livestock-census-2019.

11. **tamil_nadu_district_milk_2008_09.csv** — District-wise estimated cow and buffalo milk
    production, 30 districts, single year 2008-09 (State Handbook of Statistics 2010).
    Obtained during data collection but NOT used in this paper's analysis: it is a single
    dated (2008-09) cross-section for a state outside this paper's forecasted-growth/
    allocation set, and did not fit the Section 4.6 illustration's scope without diluting
    it. Retained here for transparency and as a starting point for future district-level
    extensions. Source: Open Government Data (OGD) Platform India (State Handbook of
    Statistics 2010 series).

All pages were retrieved on 2026-09-03.

## Known data caveats (disclosed, not hidden)

- **Telangana** was carved out of Andhra Pradesh in June 2014. NDDB's table nonetheless
  shows separately reconstructed Telangana figures back to 2009-10 (evidently a
  retrospective split by the source agency); Andhra Pradesh figures for the same years are
  NOT adjusted to exclude Telangana, so AP and Telangana series should not simply be summed
  post-2009 without double-counting awareness. The code treats Telangana and Andhra
  Pradesh as two distinct panel members throughout (no merging or summing), as stated in
  the paper.
- **Dadra & Nagar Haveli and Daman & Diu** were merged into a single UT in January 2020.
  The source table reports them as two separate short series that both terminate before
  2014-15 in parts (Daman & Diu) and continues Dadra & Nagar Haveli alone; no merged
  post-2020 series was available on the source page. We exclude this UT from the post-2019
  panel and note it as a small, low-materiality gap (it contributes <0.01% of national
  production).
- **Delhi and Ladakh** have short/interrupted series for administrative reasons (Ladakh
  became a separate UT only in 2019). These are handled as unbalanced panel entries (missing
  values kept as missing, not imputed).
- The **Livestock Census** is quinquennial (roughly every 5 years), not annual, so
  livestock-population features used in the forecasting model are the most recent available
  census figure held constant / linearly interpolated between census years, which is stated
  explicitly as a modeling simplification in the paper.

## Added in revision

12. **Revised NPDD outlay and installed BMCs** (used only to put the scenario budget in context, the paper's Sect. 3.4):
    Press Information Bureau, "Cabinet approves Revised National Program for Dairy Development (NPDD)",
    19 March 2025, https://www.pib.gov.in/PressReleseDetailm.aspx?PRID=2112791 (Rs 2,790 crore for
    2021-22 to 2025-26; 5,123 bulk milk coolers installed, 123.33 lakh litres capacity).
