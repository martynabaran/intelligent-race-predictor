# Stage 10: model for running races

129 running races of 12 athletes (runs after a ride are no longer races, see `REVIEW_RACES.md`). Metrics on total race time; `MedAPE` = median absolute percentage error. The app uses Riegel; `AutoRouter` re-selects the model inside every training fold (inner grouped CV over Riegel, Ridge, HistGB, Riegel+correction), so its scores are a nested estimate.

| protocol      | model                         |   MAPE_% all |   MedAPE_% all |   MAE_time_min all |
|:--------------|:------------------------------|-------------:|---------------:|-------------------:|
| chronological | Riegel (app)                  |         9.46 |           5.42 |               8.79 |
| chronological | Ridge (previous app model)    |         7.8  |           6.24 |               5.19 |
| chronological | HistGB                        |        11.53 |           6.64 |              12.12 |
| chronological | Riegel+correction             |         8.85 |           6.8  |               7.57 |
| chronological | AutoRouter (nested selection) |         8.85 |           6.8  |               7.57 |
| grouped-CV    | Riegel (app)                  |         6.62 |           4.57 |               4.63 |
| grouped-CV    | Ridge (previous app model)    |        10.66 |           8.45 |               6.13 |
| grouped-CV    | HistGB                        |        14.84 |          14.4  |               8.35 |
| grouped-CV    | Riegel+correction             |         6.47 |           4.96 |               4.11 |
| grouped-CV    | AutoRouter (nested selection) |         6.72 |           5.31 |               4.4  |

## MAPE per distance (grouped-CV)

| segment   |   n |   Riegel (app) |   Ridge (previous app model) |   HistGB |   Riegel+correction |   AutoRouter (nested selection) |
|:----------|----:|---------------:|-----------------------------:|---------:|--------------------:|--------------------------------:|
| marathon  |   8 |           13.2 |                         15.6 |     15.6 |                10.1 |                            11.1 |
| other     |  50 |            8   |                         10.8 |     16   |                 7.2 |                             7.5 |
| half      |   9 |            4.1 |                         10.8 |     11.7 |                 5.4 |                             6.3 |
| 5k        |  25 |            4.5 |                          8.7 |     15.1 |                 4.1 |                             4.1 |
| 10k       |  37 |            5.3 |                         10.7 |     13.6 |                 6.5 |                             6.6 |

## What AutoRouter chose (6 outer fits)

- `road`: {'Riegel+correction': 5, 'Riegel': 1}

## Conclusions

*(Written by hand from the run above; `run_modes.py` regenerates only the tables.)*

1. **Triathlon runs are no longer predicted.** 39 of the 41 detected "triathlon races" were brick training sessions
   (`REVIEW_RACES.md`), so 2 real ones remain, which is not enough for a model. Runs after a ride stay in the training
   history but are not races. The earlier "router: 9.7 → 7.9 %" gain came mostly from Ridge learning that brick runs
   are slow, so it was largely an artefact of the labels.
2. **For unseen athletes the Riegel formula is as good as anything learned.** It reaches 6.6 % MAPE (median 4.6 %,
   MAE 4.6 min). Ridge (10.7 %) and HistGB (14.8 %) are clearly worse on 129 races.
3. **Riegel+correction (Riegel × learned correction) is the only learned model that keeps up, and it wins on long
   races.** Overall it scores 6.5 % (median 5.0 %, MAE 4.1 min). Marathon goes from 13.2 to 10.1 % and other distances
   from 8.0 to 7.2 %, while half and 10 k are a bit worse. Nested selection picks it in 5 of 6 folds, at 6.7 %
   unbiased. The difference to Riegel is within noise overall but consistent on long races. It is the natural next
   candidate for the app's model, once it has an explanation panel.
4. **For athletes already in the training data, learned models help.** In the chronological protocol Ridge is best at
   7.8 % vs 9.5 % for Riegel, which supports personalisation for returning users.
5. **Label quality mattered more than model choice.** Fixing the labels changed the conclusions more than any model or
   architecture did.
