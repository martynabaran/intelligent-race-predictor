# Stage 10: model for running races

125 running races of 13 athletes (runs after a ride are no longer races, see `REVIEW_RACES.md`). Metrics on total race time; `MedAPE` = median absolute percentage error. The app uses Riegel+correction+ref; `AutoRouter` re-selects the model inside every training fold (inner grouped CV over Riegel, Ridge, HistGB, Riegel+correction with and without the reference type), so its scores are a nested estimate.

| protocol      | model                         |   MAPE_% all |   MedAPE_% all |   MAE_time_min all |
|:--------------|:------------------------------|-------------:|---------------:|-------------------:|
| chronological | Riegel                        |         7.49 |           4.66 |               7.74 |
| chronological | Ridge (previous app model)    |         8.08 |           6.55 |               5.46 |
| chronological | HistGB                        |         8.66 |           4.37 |               7.72 |
| chronological | Riegel+correction             |         7.75 |           5.31 |               6.85 |
| chronological | Riegel+correction+ref (app)   |         7.71 |           5.14 |               6.69 |
| chronological | AutoRouter (nested selection) |         7.75 |           5.31 |               6.85 |
| grouped-CV    | Riegel                        |         7.06 |           4.77 |               4.66 |
| grouped-CV    | Ridge (previous app model)    |        14.01 |          13.73 |               7.01 |
| grouped-CV    | HistGB                        |        14.91 |          12.09 |               7.93 |
| grouped-CV    | Riegel+correction             |         6.42 |           4.98 |               4.1  |
| grouped-CV    | Riegel+correction+ref (app)   |         6.37 |           4.51 |               4.04 |
| grouped-CV    | AutoRouter (nested selection) |         6.42 |           4.54 |               4.12 |

## By what Riegel projects from: a previous race or a training run

The reference is the athlete's fastest >=5 km run of the past year: a detected race for 68 races, a training run for 57.

| protocol      | model                         |   MAPE_% race |   MAPE_% training run |   MedAPE_% race |   MedAPE_% training run |
|:--------------|:------------------------------|--------------:|----------------------:|----------------:|------------------------:|
| chronological | Riegel                        |          6.01 |                 11.93 |            4.17 |                    9.74 |
| chronological | Ridge (previous app model)    |          6.67 |                 12.31 |            4.41 |                   11.13 |
| chronological | HistGB                        |          6.66 |                 14.63 |            3.6  |                   15.88 |
| chronological | Riegel+correction             |          5.59 |                 14.25 |            4.09 |                   17.87 |
| chronological | Riegel+correction+ref (app)   |          5.86 |                 13.26 |            4.14 |                   16.39 |
| chronological | AutoRouter (nested selection) |          5.59 |                 14.25 |            4.09 |                   17.87 |
| grouped-CV    | Riegel                        |          6.06 |                  8.25 |            4.05 |                    7.45 |
| grouped-CV    | Ridge (previous app model)    |         14.14 |                 13.85 |           14.49 |                   13.09 |
| grouped-CV    | HistGB                        |         16.34 |                 13.2  |           11.96 |                   12.34 |
| grouped-CV    | Riegel+correction             |          5.6  |                  7.4  |            4.18 |                    5.32 |
| grouped-CV    | Riegel+correction+ref (app)   |          5.72 |                  7.15 |            4.5  |                    4.54 |
| grouped-CV    | AutoRouter (nested selection) |          5.62 |                  7.37 |            4.25 |                    4.96 |

## MAPE per distance (grouped-CV)

| segment   |   n |   Riegel |   Ridge (previous app model) |   HistGB |   Riegel+correction |   Riegel+correction+ref (app) |   AutoRouter (nested selection) |
|:----------|----:|---------:|-----------------------------:|---------:|--------------------:|------------------------------:|--------------------------------:|
| marathon  |   8 |     13.2 |                         12.1 |     15.4 |                11   |                          10.3 |                            10.9 |
| half      |  12 |      8.2 |                         11.4 |     10.5 |                 7.6 |                           8.2 |                             8.2 |
| other     |  30 |      9.1 |                         15.5 |     15.3 |                 6.2 |                           5.9 |                             5.8 |
| 5k        |  34 |      4.9 |                         14.9 |     15.5 |                 4.8 |                           4.7 |                             5   |
| 10k       |  41 |      5.8 |                         13.2 |     15.4 |                 6.7 |                           6.8 |                             6.7 |

## What AutoRouter chose (6 outer fits)

- `road`: {'Riegel+correction': 3, 'Riegel+correction+ref': 3}

## Conclusions

*(Written by hand from the run above; `run_modes.py` regenerates only the tables, and keeps this
section.)* The data are the 125 running races left after the manual review (`REVIEW_RACES.md`,
`data/race_labels.csv`): the detector's false races were removed, the 33 races it only finds with
the heart-rate spike filter were added, and runs after a ride are not road races to predict.

1. **Runs after a ride are not predicted as road races.** The review of the 39 weakest such
   candidates found 27 brick training sessions and 12 real races, so the detector cannot tell them
   apart. The real ones are triathlon run legs: slower than road races and too few for a model of
   their own. They stay in the training history.
2. **Riegel is only as good as its reference run.** Its reference is the fastest ≥5 km run of the past
   year. From a race (68 races) it errs 6.1 % on unseen athletes; from a training run (57) it errs
   8.3 % (chronological protocol: 6.0 % vs 11.9 %). This reconciles our results with the thesis, which
   found Riegel weak (NRMSE 18 %) projecting Kaggle training runs to the marathon.
3. **The app uses Riegel × a learned correction that knows the reference type
   (`Riegel+correction+ref`).** On unseen athletes it reaches **6.4 % MAPE** (median 4.5 %, MAE
   4.0 min) against 7.1 % for plain Riegel; the gain is largest when the reference is a training run
   (7.2 % vs 8.3 %) and small when it is a race (5.7 % vs 6.1 %). Nested selection picked this model
   in 3 of 6 folds and the correction without the reference type in the other 3 (6.4 % either way),
   so the two are tied and 6.4 % is the unbiased estimate.
4. **By distance** (unseen athletes), the correction helps on the marathon (13.2 → 10.3 %, only 8
   races) and on non-standard distances (9.1 → 5.9 %, 30 races) and is slightly worse on 10 km
   (5.8 → 6.8 %, 41 races); on 5 km and the half marathon it is about equal.
5. **No learned model beats Riegel on known athletes either.** In the chronological protocol Riegel
   scores 7.5 %, the app model 7.7 %, Ridge 8.1 % and gradient boosting 8.7 %. An earlier version of
   this report (129 races, before the review) found learned models better for returning users;
   with the cleaned labels that advantage is gone, and personalisation does not pay at this data
   size.
6. **The label cleanup changed the ranking of the learned models, not of the Riegel-based ones.**
   Ridge went from 10.7 % to 14.0 % on unseen athletes and gradient boosting from 14.8 % to 14.9 %,
   while plain Riegel moved from 6.6 % to 7.1 % and the app model from 6.1 % to 6.4 %. The learned
   tabular models do not generalise to new athletes from 13 athletes' races.
7. **Race detection runs on the user's export too** (`cleaning_pipeline("predict")`), so the app
   knows whether the reference was a race. *(Figures from the run before the reviewed labels, not
   re-measured:* on the sample athletes, train-time and predict-time labels agreed on all but 2 of
   ~3,600 shared runs; runs without heart rate exist only at predict time and added 14 races.)
   The reviewed labels exist for the training data only: a new user's export is labelled by the
   detector, which the review found right about half the time on its weakest cases.
