# Hyper-parameter tuning (Optuna)

> **Note:** computed on the earlier 170-race set, before the manual race review (`REVIEW_RACES.md`); to be recomputed on the reviewed labels in stage 15.

170 races, 14 athletes.

Grouped (by athlete) CV. 'nested' re-tunes inside every outer training fold, so it is an unbiased estimate; the single study below is tuned on all folds and therefore optimistic.

| model                    |   MAPE_% |   MAE_time_min |
|:-------------------------|---------:|---------------:|
| HistGB default           |    13.59 |           6.86 |
| HistGB tuned (nested CV) |    13.32 |           6.72 |

Best params on all data: `{'learning_rate': 0.05569770414467638, 'max_iter': 150, 'max_leaf_nodes': 25, 'min_samples_leaf': 10, 'l2_regularization': 0.05049252962834131}` (inner CV MAPE 13.40%)
