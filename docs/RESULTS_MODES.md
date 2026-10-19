# Stage 10: model for running races

129 running races of 12 athletes (runs after a ride are no longer races, see `REVIEW_RACES.md`). Metrics on total race time; `MedAPE` = median absolute percentage error. The app uses Riegel+correction; `AutoRouter` re-selects the model inside every training fold (inner grouped CV over Riegel, Ridge, HistGB, Riegel+correction), so its scores are a nested estimate.

| protocol      | model                         |   MAPE_% all |   MedAPE_% all |   MAE_time_min all |
|:--------------|:------------------------------|-------------:|---------------:|-------------------:|
| chronological | Riegel                        |         9.46 |           5.42 |               8.79 |
| chronological | Ridge (previous app model)    |         7.8  |           6.24 |               5.19 |
| chronological | HistGB                        |        11.53 |           6.64 |              12.12 |
| chronological | Riegel+correction (app)       |         8.85 |           6.8  |               7.57 |
| chronological | AutoRouter (nested selection) |         8.85 |           6.8  |               7.57 |
| grouped-CV    | Riegel                        |         6.62 |           4.57 |               4.63 |
| grouped-CV    | Ridge (previous app model)    |        10.66 |           8.45 |               6.13 |
| grouped-CV    | HistGB                        |        14.84 |          14.4  |               8.35 |
| grouped-CV    | Riegel+correction (app)       |         6.47 |           4.96 |               4.11 |
| grouped-CV    | AutoRouter (nested selection) |         6.72 |           5.31 |               4.4  |

## By what Riegel projects from: a previous race or a training run

The reference is the athlete's fastest >=5 km run of the past year: a detected race for 85 races, a training run for 44.

| protocol      | model                         |   MAPE_% race |   MAPE_% training run |   MedAPE_% race |   MedAPE_% training run |
|:--------------|:------------------------------|--------------:|----------------------:|----------------:|------------------------:|
| chronological | Riegel                        |          3.49 |                 17.83 |            3.17 |                   18.4  |
| chronological | Ridge (previous app model)    |          6.13 |                 10.14 |            3.91 |                    9.76 |
| chronological | HistGB                        |          4    |                 22.08 |            2.86 |                   22.48 |
| chronological | Riegel+correction (app)       |          4.15 |                 15.45 |            3.14 |                   15.41 |
| chronological | AutoRouter (nested selection) |          4.15 |                 15.45 |            3.14 |                   15.41 |
| grouped-CV    | Riegel                        |          5.72 |                  8.35 |            4.38 |                    6.04 |
| grouped-CV    | Ridge (previous app model)    |         10.03 |                 11.89 |            8.45 |                    8.06 |
| grouped-CV    | HistGB                        |         16.79 |                 11.07 |           16.22 |                    7.03 |
| grouped-CV    | Riegel+correction (app)       |          5.35 |                  8.65 |            4.11 |                    6.92 |
| grouped-CV    | AutoRouter (nested selection) |          5.58 |                  8.92 |            4.64 |                    6.92 |

## MAPE per distance (grouped-CV)

| segment   |   n |   Riegel |   Ridge (previous app model) |   HistGB |   Riegel+correction (app) |   AutoRouter (nested selection) |
|:----------|----:|---------:|-----------------------------:|---------:|--------------------------:|--------------------------------:|
| marathon  |   8 |     13.2 |                         15.6 |     15.6 |                      10.1 |                            11.1 |
| other     |  50 |      8   |                         10.8 |     16   |                       7.2 |                             7.5 |
| half      |   9 |      4.1 |                         10.8 |     11.7 |                       5.4 |                             6.3 |
| 5k        |  25 |      4.5 |                          8.7 |     15.1 |                       4.1 |                             4.1 |
| 10k       |  37 |      5.3 |                         10.7 |     13.6 |                       6.5 |                             6.6 |

## What AutoRouter chose (6 outer fits)

- `road`: {'Riegel+correction': 5, 'Riegel': 1}

## Conclusions

*(Written by hand from the run above; `run_modes.py` regenerates only the tables.)*

1. **Triathlon runs are no longer predicted.** 39 of the 41 detected "triathlon races" were brick training sessions
   (`REVIEW_RACES.md`), so 2 real ones remain. Runs after a ride stay in the training history. The earlier
   "router: 9.7 → 7.9 %" gain (#6) came mostly from Ridge learning that brick runs are slow.
2. **The app uses Riegel × learned correction.** On unseen athletes it reaches 6.5 % MAPE (MAE 4.1 min), vs 6.6 %
   (MAE 4.6 min) for plain Riegel. Nested selection picks it in 5 of 6 folds (6.7 % unbiased). It is better on long
   races (marathon 13.2 → 10.1 %, other distances 8.0 → 7.2 %) and a little worse on half and 10 k. On the sample
   exports it changes predictions by −0.7 % (5 k) to +3.5 % (marathon) and slightly narrows the ranges.
3. **Riegel is only as good as its reference run.** Its reference is the fastest ≥5 km run of the past year. When that
   run was a race (85 races), Riegel errs 5.7 % on unseen athletes. When it was a training run (44 races), it errs
   8.4 % on unseen athletes and **17.8 %** on known athletes' latest races. This reconciles our results with the
   thesis, which found Riegel weak (NRMSE 18 %) projecting Kaggle training runs to the marathon. Riegel is good at
   what it was made for (race → race over similar distances) and weak at projecting training runs, especially to
   long races.
4. **Next step: a reference-aware correction.** Adding "was the reference a race" to the correction gives the best
   result so far on unseen athletes: **6.1 %** overall, 7.7 % with a training-run reference, 9.6 % on the marathon.
   This was a single run, not nested yet. It needs race detection on the user's own export at prediction time, which
   the app does not run today.
5. **For athletes already in the training data, learned models win** (chronological: Ridge 7.8 %). This supports
   personalisation for returning users.
6. **Label quality mattered more than model choice.**
