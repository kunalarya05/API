# E-Waste ML Price Engine — Stress-Test Report

This report records the original pre-fix run. The later ML adapter enforces
informal <= mandi <= authorized before returning direct ML quotes. The known
per-piece Delhi/Keypad Phone crossing is included in the compatibility test;
the full original randomized stress suite has not been re-run here.

## Executive result
- **Total checks:** 6,194
- **Passed:** 6,158
- **Failed:** 36
- **Pass rate:** 99.419%
- **Randomized batch fuzz cases:** 5,000

## Exhaustive model-layer coverage
- Per-kg exact subcategory × known-city rows: 624
- Per-piece exact subcategory × known-city rows: 156
- Per-kg channel-order violations: 0
- Per-piece channel-order violations: 1

## Test groups
| group              |   tests |   passed |   failed |   pass_rate_percent |
|:-------------------|--------:|---------:|---------:|--------------------:|
| aliases            |       7 |        7 |        0 |             100     |
| date_extrapolation |      48 |       48 |        0 |             100     |
| exhaustive_batch   |     780 |      779 |        1 |              99.872 |
| fuzz_batch         |    5000 |     4965 |       35 |              99.3   |
| scaling            |      32 |       32 |        0 |             100     |
| unseen_geography   |      32 |       32 |        0 |             100     |
| validation         |       7 |        7 |        0 |             100     |
| wrapper_broad      |      96 |       96 |        0 |             100     |
| wrapper_exact      |     192 |      192 |        0 |             100     |

## What was tested
- Every supported exact subcategory across every known city in a batched model-layer sweep.
- All three price outputs: informal, mandi, authorized.
- Broad-category wrapper quotes across every category and known city.
- Exact-subcategory wrapper quotes across every category and known city.
- Per-kg/per-piece lot-value scaling from tiny to very large lots.
- Prediction intervals.
- Channel ordering: informal <= mandi <= authorized.
- Date extrapolation from 2020 through 2040.
- Unseen cities and states.
- Category aliases.
- Rejection of invalid categories, subcategories, channels, negative values, and missing required quantity/weight.
- 5,000 randomized out-of-distribution/fuzz feature combinations.

## Interpretation
This is a **software robustness and inference stress test**. It is different from predictive
accuracy validation. Predictive quality should still be reported with MAE/RMSE/R²/MAPE on
a properly held-out real historical test set.
