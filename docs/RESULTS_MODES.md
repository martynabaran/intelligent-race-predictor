# Stage 10: model for running races

129 running races of 12 athletes (runs after a ride are no longer races, see `REVIEW_RACES.md`). Metrics on total race time; `MedAPE` = median absolute percentage error. The app uses Riegel+correction+ref; `AutoRouter` re-selects the model inside every training fold (inner grouped CV over Riegel, Ridge, HistGB, Riegel+correction with and without the reference type), so its scores are a nested estimate.

| protocol      | model                         |   MAPE_% all |   MedAPE_% all |   MAE_time_min all |
|:--------------|:------------------------------|-------------:|---------------:|-------------------:|
| chronological | Riegel                        |         9.46 |           5.42 |               8.79 |
| chronological | Ridge (previous app model)    |         7.8  |           6.24 |               5.19 |
| chronological | HistGB                        |        11.53 |           6.64 |              12.12 |
| chronological | Riegel+correction             |         8.85 |           6.8  |               7.57 |
| chronological | Riegel+correction+ref (app)   |         9.33 |           8.29 |               8.13 |
| chronological | AutoRouter (nested selection) |         9.33 |           8.29 |               8.13 |
| grouped-CV    | Riegel                        |         6.62 |           4.57 |               4.63 |
| grouped-CV    | Ridge (previous app model)    |        10.66 |           8.45 |               6.13 |
| grouped-CV    | HistGB                        |        14.84 |          14.4  |               8.35 |
| grouped-CV    | Riegel+correction             |         6.47 |           4.96 |               4.11 |
| grouped-CV    | Riegel+correction+ref (app)   |         6.12 |           4.91 |               3.91 |
| grouped-CV    | AutoRouter (nested selection) |         6.12 |           4.91 |               3.91 |

## By what Riegel projects from: a previous race or a training run

The reference is the athlete's fastest >=5 km run of the past year: a detected race for 85 races, a training run for 44.

| protocol      | model                         |   MAPE_% race |   MAPE_% training run |   MedAPE_% race |   MedAPE_% training run |
|:--------------|:------------------------------|--------------:|----------------------:|----------------:|------------------------:|
| chronological | Riegel                        |          3.49 |                 17.83 |            3.17 |                   18.4  |
| chronological | Ridge (previous app model)    |          6.13 |                 10.14 |            3.91 |                    9.76 |
| chronological | HistGB                        |          4    |                 22.08 |            2.86 |                   22.48 |
| chronological | Riegel+correction             |          4.15 |                 15.45 |            3.14 |                   15.41 |
| chronological | Riegel+correction+ref (app)   |          5.16 |                 15.18 |            4.27 |                   16.73 |
| chronological | AutoRouter (nested selection) |          5.16 |                 15.18 |            4.27 |                   16.73 |
| grouped-CV    | Riegel                        |          5.72 |                  8.35 |            4.38 |                    6.04 |
| grouped-CV    | Ridge (previous app model)    |         10.03 |                 11.89 |            8.45 |                    8.06 |
| grouped-CV    | HistGB                        |         16.79 |                 11.07 |           16.22 |                    7.03 |
| grouped-CV    | Riegel+correction             |          5.35 |                  8.65 |            4.11 |                    6.92 |
| grouped-CV    | Riegel+correction+ref (app)   |          5.28 |                  7.74 |            4.36 |                    5.43 |
| grouped-CV    | AutoRouter (nested selection) |          5.28 |                  7.74 |            4.36 |                    5.43 |

## MAPE per distance (grouped-CV)

| segment   |   n |   Riegel |   Ridge (previous app model) |   HistGB |   Riegel+correction |   Riegel+correction+ref (app) |   AutoRouter (nested selection) |
|:----------|----:|---------:|-----------------------------:|---------:|--------------------:|------------------------------:|--------------------------------:|
| marathon  |   8 |     13.2 |                         15.6 |     15.6 |                10.1 |                           9.6 |                             9.6 |
| other     |  50 |      8   |                         10.8 |     16   |                 7.2 |                           6.7 |                             6.7 |
| half      |   9 |      4.1 |                         10.8 |     11.7 |                 5.4 |                           5.3 |                             5.3 |
| 5k        |  25 |      4.5 |                          8.7 |     15.1 |                 4.1 |                           4.1 |                             4.1 |
| 10k       |  37 |      5.3 |                         10.7 |     13.6 |                 6.5 |                           6.2 |                             6.2 |

## What AutoRouter chose (6 outer fits)

- `road`: {'Riegel+correction+ref': 6}

## Conclusions

*(Written by hand from the run above; `run_modes.py` regenerates only the tables.)*

1. **Triathlon runs are no longer predicted.** 39 of the 41 detected "triathlon races" were brick training sessions
   (`REVIEW_RACES.md`), so 2 real ones remain. Runs after a ride stay in the training history. The earlier
   "router: 9.7 → 7.9 %" gain (#6) came mostly from Ridge learning that brick runs are slow.
2. **Riegel is only as good as its reference run.** Its reference is the fastest ≥5 km run of the past year. From a
   race (85 races) it errs 5.7 % on unseen athletes. From a training run (44) it errs 8.4 %, and 17.8 % on known
   athletes' latest races. This reconciles our results with the thesis, which found Riegel weak (NRMSE 18 %)
   projecting Kaggle training runs to the marathon.
3. **The app uses Riegel × a learned correction that knows the reference type (`Riegel+correction+ref`).** On unseen
   athletes it reaches **6.1 % MAPE** (median 4.9 %, MAE 3.9 min), vs 6.5 % for the correction without the
   reference type and 6.6 % for plain Riegel. It helps most where Riegel is weak: 7.7 % (vs 8.4 %) from a training
   run. Nested selection picked it in **6 of 6** folds, so 6.1 % is the unbiased estimate.
4. **The cost is on known athletes.** In the chronological protocol it scores 9.3 % (vs 8.9 % without the reference
   type, and Ridge 7.8 %). The app predicts for new users, but returning users would be better served by
   personalisation.
5. **Race detection now runs on the user's export too** (`cleaning_pipeline("predict")`), so the app knows whether the
   reference was a race. On the sample athletes, train-time and predict-time labels agree on all but 2 of ~3,600
   shared runs. Runs without heart rate exist only at predict time and add 14 races detected by pace or distance.
   On the sample exports, predictions change by −2.8 % to +0.9 % vs the previous correction: faster when the
   reference is a race, slower when it is a training run. The median range width goes from 1.23 to 1.20.
6. **Label quality mattered more than model choice.**
