# Walmart Sales Forecasting

Forecasting weekly sales by store and department for Walmart, using the [Walmart Recruiting - Store Sales Forecasting dataset](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) (45 stores, 81 departments, Feb 2010–Oct 2012).

## Business Problem

Accurate short-term sales forecasts help retailers plan staffing, inventory, and promotions. This project builds a forecasting model at the store-department-week level, framed as an **operational forecasting problem**: given known store characteristics, planned promotional markdowns, and prevailing economic conditions, predict next week's sales. (See the note on data leakage below for why this framing matters.)

## Data Cleaning Notes

- Merged 4 source files (`train`, `test`, `stores`, `features`) into one dataset joined on `Store` and `Date`.
- **Excluded Department 47** after investigation: it showed a 39% negative-sales rate (vs. <2% for nearly every other department) and inconsistent, erratic patterns across stores with no learnable trend — evidence pointed to a data-quality anomaly rather than genuine demand, so it was excluded and documented rather than silently dropped or blindly kept.
- **Diagnosed two distinct missingness patterns** in `features.csv` and handled each differently:
  - `MarkDown1-5`: missing entirely in 2010, mostly missing in 2011, complete by 2012 — consistent with the promotional tracking program simply not existing yet early on. Filled with 0.
  - `CPI`/`Unemployment`: missing only in the last 3 months of the dataset — consistent with real reporting lag in official economic statistics. Forward-filled per store, since these values change slowly and a recent true value is a much better estimate than 0.
- **Found and corrected a labeling error in `IsHoliday`**: the dataset's holiday flag lands on the week of Dec 31 (a low-sales week) rather than the week of Dec 24 (the actual highest-selling week of the year, both years in the data). Added a corrected `IsHoliday_corrected` flag that includes the true pre-Christmas peak week, while preserving the original flags for Super Bowl, Labor Day, and Thanksgiving.

## A Note on Data Leakage

Promotional markdowns are typically planned in advance by the retailer, not an unknowable future event — so using markdown values for the forecast period is legitimate under an **operational forecasting** framing ("given our known promotional calendar, what do we expect sales to be"), and is *not* the same kind of leakage as using, say, future actual sales. This distinction was made explicitly rather than assumed, and would need revisiting if the goal were pure historical-pattern-only forecasting with no external inputs.

## Feature Engineering

- Calendar features: `Year`, `Month`, `WeekOfYear`
- `Sales_Lag1` — previous week's sales (short-term momentum)
- `Sales_Lag52` — same week, one year prior (direct yearly seasonality signal)
- `Sales_RollingMean4` — trailing 4-week average, computed only on prior weeks to avoid leakage

**Tradeoff decision:** requiring a full year of history for `Sales_Lag52` reduces the usable dataset from ~421K to ~261K rows (dropping the earliest, least mature period). This was tested and found worthwhile — removing `Sales_Lag52` increased validation error from $1,440 to $1,574 MAE, confirming it as a high-value feature worth the data loss. A hybrid short/long-history model approach is noted below as a future improvement.

## Train / Validation / Test Split

Unlike a typical classification problem, this data was split **chronologically**, not randomly, to avoid leaking future information into training — a random shuffle would let the model "see" future weeks during training, an easy and common mistake in time-series projects:

- **Train:** Feb 2011 – Apr 2012 (63 weeks)
- **Validation:** Apr – Jul 2012 (14 weeks) — used for model comparison and hyperparameter tuning only
- **Test:** Jul – Oct 2012 (14 weeks) — held out completely until final evaluation

## Modeling Approach

**Baseline:** a naive "predict last week's value" forecast, MAE $1,506 on the test set (~9-10% of average weekly sales) — a surprisingly strong baseline, reflecting real week-to-week sales momentum, and the bar any real model needs to clear.

Given weak linear correlations found during EDA (even markdowns showed near-zero linear correlation with sales, both overall and split by holiday period) but plausible non-linear/interaction effects, tree-based models were used directly rather than starting from a linear baseline.

**Random Forest was tuned via grid search on the validation set**, testing depth and leaf-size constraints. Performance consistently improved as trees were constrained (shallower depth, larger minimum leaf size) up to a point, then got worse beyond it — confirming a genuine sweet spot rather than a one-directional trend:

| Model | Validation MAE | vs. naive |
|---|---|---|
| Naive baseline | 1,662.72 | — |
| Untuned Random Forest | 1,440.20 | 13.4% better |
| **Tuned Random Forest** (400 trees, depth 12, min leaf 8) | **1,385.55** | **16.7% better** |

## Final Result (Held-Out Test Set)

| Metric | Value |
|---|---|
| Test MAE | **$1,274.79** |
| Test MAE as % of average weekly sales | **8.0%** |
| Improvement over naive baseline (test set) | **15.4%** |

## Feature Importance

`Sales_Lag52` (same week last year) dominates the model, accounting for ~81% of total feature importance, with `Sales_Lag1` contributing another ~13%. Together these two features explain over 95% of the model's decisions — consistent with the strong, consistent year-over-year seasonality found during EDA. Removing `Sales_Lag52` was tested directly (not just inferred from the importance score) and confirmed a real, substantial contribution: MAE rose from $1,440 to $1,574 without it.

All other features — including store size, markdowns, temperature, CPI, and the corrected holiday flag — individually contribute less than 0.2% importance each. This doesn't necessarily mean they're useless (linear correlation checks during EDA suggested possible non-linear or interaction effects the model may still be partially capturing), but their marginal individual contribution is small compared to the lag features.

## Repository Structure

```
├── README.md
└── sales_forecasting.ipynb   # Full data cleaning, EDA, feature engineering, and modeling notebook
```

## Tools Used

Python, pandas, scikit-learn (RandomForestRegressor), matplotlib

## Limitations & Next Steps

- **No holiday period in the test set:** due to the 91 weeks of usable data after the lag-feature cutoff, the test window (Jul–Oct 2012) doesn't include a December/holiday period, so model performance on the year's most extreme seasonal spike remains unvalidated.
- **Hybrid short/long-history model:** a two-model system — one relying on `Sales_Lag52` for established store-departments, one without it for newer combinations lacking a full year of history — was considered but not built. Initial checks suggest the current test period may not contain many rows actually affected by this (most store-departments already have a full year of history by mid-2012), making this a lower-priority improvement for this specific dataset, though it would matter more in a live production setting with continuously new store-department combinations.
- **Markdown features remain weakly linearly correlated with sales** even after splitting by holiday period; further investigation (e.g., explicit interaction terms with store size/type) could clarify whether they're truly low-value or whether their effect is being masked by simple correlation analysis.
