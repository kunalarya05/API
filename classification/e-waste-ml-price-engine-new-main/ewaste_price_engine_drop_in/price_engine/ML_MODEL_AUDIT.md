# ML Pricing Model Audit

## Source data
- Price rows: 1,116
- Material rows: 65
- Categories: 8
- Locations/cities: 12
- Dates: 5
- Per-kg rows: 960
- Per-piece rows: 156

The 65 curated material baselines contain market-source URLs and observation dates:
34 `DIRECT`, 24 `DIRECT_RANGE`, and 7 `PROXY` evidence rows. These are the
source-backed inputs to the expanded price table, not 65 independently observed
quotes for each location and date.

## Synthetic flag
is_synthetic
True    1116

All 1,116 generated location/date/channel price rows are marked synthetic. The
overall data provenance is hybrid: source-backed material baselines plus
synthetically expanded pricing scenarios. Scores below describe the generated
scenarios and must not be reported as verified real-market predictive accuracy.

## Modeling design
- Models compared: Linear Regression, Random Forest, XGBoost
- Validation: 25% of cities held out as complete groups
- Selection metric: MAE
- Separate model family by pricing unit
- Separate model per market channel
- Deployment models refit on all available rows after evaluation

## Leakage controls
The following columns were not used as model features:
- source_base_price_min_inr
- source_base_price_max_inr
- source_base_price_mid_inr
- location_price_factor
- category_location_factor
- date_factor
- market_range_min_inr
- market_range_max_inr
- formal_channel_bonus_inr
- formal_bonus_percentage
- doorstep_informal_price_inr
- wholesale_mandi_price_inr
- authorized_recycler_offered_price_inr

## Best model per task
| unit      | channel    | model         |   mae_inr |   rmse_inr |       r2 |   mape_percent |
|:----------|:-----------|:--------------|----------:|-----------:|---------:|---------------:|
| per_kg    | authorized | random_forest |   8.97901 |    22.0079 | 0.997002 |        2.26299 |
| per_kg    | informal   | random_forest |   6.5242  |    14.6873 | 0.997879 |        2.17968 |
| per_kg    | mandi      | random_forest |   8.43634 |    19.5908 | 0.997238 |        2.29258 |
| per_piece | authorized | xgboost       |   6.27038 |     9.8034 | 0.999777 |        2.6505  |
| per_piece | informal   | xgboost       |   5.8747  |    10.2018 | 0.999617 |        2.92949 |
| per_piece | mandi      | xgboost       |   7.08661 |    11.7151 | 0.999625 |        3.00878 |

## Interpretation
Use `model_comparison.csv` for all benchmark results and
`holdout_predictions.csv` for row-level held-out errors.
