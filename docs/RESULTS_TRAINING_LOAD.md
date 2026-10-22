# Stage 9c: training-load features and the Banister model

125 races (0 triathlon runs), 13 athletes. Load features: `load_ctl, load_atl, load_tsb, load_ctl_ramp_4w` (state on the eve of the race; TRIMP load, CTL 42 d, ATL 7 d). Metrics on total race time. `grouped-CV` holds out whole athletes; `chronological` holds out each athlete's latest races.

## 1. Do load features improve the models?

MAPE (%) per segment, the same model with and without the load features.

| protocol      | model                     | load         |   MAPE_% all |   MAPE_% stand-alone |
|:--------------|:--------------------------|:-------------|-------------:|---------------------:|
| chronological | Riegel (ignores features) | without load |         7.49 |                 7.49 |
| chronological | Riegel (ignores features) | with load    |         7.49 |                 7.49 |
| chronological | Ridge                     | without load |         8.08 |                 8.08 |
| chronological | Ridge                     | with load    |         7.93 |                 7.93 |
| chronological | HistGB                    | without load |         8.66 |                 8.66 |
| chronological | HistGB                    | with load    |         8.61 |                 8.61 |
| chronological | Router (app model)        | without load |         7.71 |                 7.71 |
| chronological | Router (app model)        | with load    |         7.71 |                 7.71 |
| grouped-CV    | Riegel (ignores features) | without load |         7.06 |                 7.06 |
| grouped-CV    | Riegel (ignores features) | with load    |         7.06 |                 7.06 |
| grouped-CV    | Ridge                     | without load |        14.01 |                14.01 |
| grouped-CV    | Ridge                     | with load    |        14.46 |                14.46 |
| grouped-CV    | HistGB                    | without load |        14.91 |                14.91 |
| grouped-CV    | HistGB                    | with load    |        15.08 |                15.08 |
| grouped-CV    | Router (app model)        | without load |         6.37 |                 6.37 |
| grouped-CV    | Router (app model)        | with load    |         6.37 |                 6.37 |

### Paired change in absolute percentage error (grouped-CV)

Mean change per race in percentage points when load features are added (negative = better) with a 95 % interval from resampling athletes. `share_improved` is the fraction of races whose error went down.

| model              | segment     |   n |   mean_change_pp |   ci_low_pp |   ci_high_pp |   share_improved |
|:-------------------|:------------|----:|-----------------:|------------:|-------------:|-----------------:|
| Ridge              | all         | 125 |             0.46 |       -0.39 |         1.34 |             0.46 |
| Ridge              | stand-alone | 125 |             0.46 |       -0.39 |         1.34 |             0.46 |
| HistGB             | all         | 125 |             0.17 |       -0.77 |         1.14 |             0.44 |
| HistGB             | stand-alone | 125 |             0.17 |       -0.77 |         1.14 |             0.44 |
| Router (app model) | all         | 125 |             0    |        0    |         0    |             0    |
| Router (app model) | stand-alone | 125 |             0    |        0    |         0    |             0    |

## 2. How well does the Banister model describe form?

Per athlete with at least 5 race days: each race is forecast from a fit on the earlier races only. Performance = Riegel-equivalent 10 km speed; errors are relative (%). Baselines: running mean of earlier races, previous race. `at_bound` = a fitted time constant sits on its limit (not identified by the data).

| athlete   |   races | method     |   tau1 |   tau2 |   forecasts | at_bound   |   banister_MAPE_% |   running_mean_MAPE_% |   previous_race_MAPE_% |
|:----------|--------:|:-----------|-------:|-------:|------------:|:-----------|------------------:|----------------------:|-----------------------:|
| strava_11 |      21 | full       |  80    |   3.8  |          18 | True       |             13.8  |                 13.09 |                  13.26 |
| strava_12 |      15 | full       |  80    |   3    |          12 | True       |              8.97 |                  8.13 |                   5.76 |
| strava_13 |      23 | full       |  23.75 |  19    |          20 | True       |             10.02 |                 10.19 |                   5.95 |
| strava_14 |      16 | full       |  80    |  19    |          13 | True       |              6.89 |                  5.19 |                   5.47 |
| strava_15 |      13 | full       |  20    |   7.08 |          10 | True       |              7.88 |                 10.63 |                  11.28 |
| strava_16 |       8 | full       |  80    |  19    |           5 | True       |              6.65 |                  6.46 |                   4.77 |
| strava_18 |       6 | fixed_taus |  42    |   7    |           3 | False      |             11.3  |                  7.12 |                  12.35 |
| strava_7  |      13 | full       |  80    |   4.89 |          10 | True       |              6.34 |                  6.46 |                   3.21 |

Pooled over 91 forecasts (8 athletes): Banister **9.40 %**, running mean **9.11 %**, previous race **7.73 %**. 7 of 7 full fits have a time constant at a bound.

## Conclusions

Written by hand from the tables above; the numbers are the script's. The data are the 125 running
races of the reviewed labels (see `RESULTS_MODES.md`). The first version of this report used the 170
unreviewed races; the conclusions did not change.

1. **Load features do not improve prediction.** On unseen athletes Ridge changes by +0.46 pp (95 %
   interval −0.39 to 1.34) and gradient boosting by +0.17 pp (−0.77 to 1.14): no reliable effect,
   and the point estimates are on the wrong side. The app's model is unchanged because Riegel
   ignores the features. (On the unreviewed 170 races Ridge was slightly *worse*, +0.46 pp with an
   interval that excluded 0.) The features stay **off** (`include_load=False`) in every model.
2. **The Banister model does not describe form better than naive forecasts.** Pooled over 91
   out-of-sample forecasts for 8 athletes it errs 9.4 %, against 9.1 % for the running mean of
   earlier races and 7.7 % for simply repeating the previous race. All 7 full fits have a time
   constant stuck at a bound, i.e. the data cannot identify them with 6 to 23 races per athlete.
   The code is correct (it recovers known parameters on synthetic data, `tests/test_banister.py`),
   so the limit is the data, not the implementation.
3. **Label noise is not the explanation.** The review cleaned the race labels and the result held.
   Remaining suspects, none of them tested here: heart rate is a noisy proxy for load; 13 athletes
   are too few for per-athlete physiology; training-time and query-time load differ slightly
   (training drops activities without valid heart rate, prediction imputes it); the first ~6 weeks
   of every history underestimate fitness.
4. **What stays useful:** CTL / ATL / TSB are a sound, descriptive picture of training (the
   *Training load* page of the app) and the building blocks the planner will need. They are not a
   proven predictor of race time here.
5. **Consequences for later stages.** The what-if stage (13) should not promise model-based form
   effects from Banister; plan rules (stage 12) do not depend on it. Revisit load features if the
   data grow: it is a one-line change (`ds.with_load()`).
