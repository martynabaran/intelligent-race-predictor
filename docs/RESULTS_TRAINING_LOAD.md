# Stage 9c: training-load features and the Banister model

170 races (41 triathlon runs), 14 athletes. Load features: `load_ctl, load_atl, load_tsb, load_ctl_ramp_4w` (state on the eve of the race; TRIMP load, CTL 42 d, ATL 7 d). Metrics on total race time. `grouped-CV` holds out whole athletes; `chronological` holds out each athlete's latest races.

## 1. Do load features improve the models?

MAPE (%) per segment, the same model with and without the load features.

| protocol      | model                     | load         |   MAPE_% all |   MAPE_% stand-alone |   MAPE_% triathlon |
|:--------------|:--------------------------|:-------------|-------------:|---------------------:|-------------------:|
| chronological | Riegel (ignores features) | without load |        11.04 |                 8.38 |              20.52 |
| chronological | Riegel (ignores features) | with load    |        11.04 |                 8.38 |              20.52 |
| chronological | Ridge                     | without load |         9.79 |                10.02 |               8.94 |
| chronological | Ridge                     | with load    |         9.96 |                10.48 |               8.09 |
| chronological | HistGB                    | without load |         7.1  |                 7.45 |               5.85 |
| chronological | HistGB                    | with load    |         8.2  |                 8.77 |               6.17 |
| chronological | Router (app model)        | without load |         8.51 |                 8.38 |               8.94 |
| chronological | Router (app model)        | with load    |         8.32 |                 8.38 |               8.09 |
| grouped-CV    | Riegel (ignores features) | without load |        10.58 |                 6.62 |              23.03 |
| grouped-CV    | Riegel (ignores features) | with load    |        10.58 |                 6.62 |              23.03 |
| grouped-CV    | Ridge                     | without load |         9.73 |                 8.97 |              12.12 |
| grouped-CV    | Ridge                     | with load    |        10.11 |                 9.43 |              12.23 |
| grouped-CV    | HistGB                    | without load |        13.97 |                14.54 |              12.2  |
| grouped-CV    | HistGB                    | with load    |        14.07 |                14.66 |              12.24 |
| grouped-CV    | Router (app model)        | without load |         7.94 |                 6.62 |              12.12 |
| grouped-CV    | Router (app model)        | with load    |         7.97 |                 6.62 |              12.23 |

### Paired change in absolute percentage error (grouped-CV)

Mean change per race in percentage points when load features are added (negative = better) with a 95 % interval from resampling athletes. `share_improved` is the fraction of races whose error went down.

| model              | segment     |   n |   mean_change_pp |   ci_low_pp |   ci_high_pp |   share_improved |
|:-------------------|:------------|----:|-----------------:|------------:|-------------:|-----------------:|
| Ridge              | all         | 170 |             0.38 |        0.1  |         0.67 |             0.35 |
| Ridge              | stand-alone | 129 |             0.46 |        0.05 |         0.82 |             0.33 |
| Ridge              | triathlon   |  41 |             0.12 |       -0.8  |         0.43 |             0.39 |
| HistGB             | all         | 170 |             0.1  |       -0.21 |         0.45 |             0.52 |
| HistGB             | stand-alone | 129 |             0.12 |       -0.29 |         0.88 |             0.55 |
| HistGB             | triathlon   |  41 |             0.04 |       -0.32 |         1.36 |             0.44 |
| Router (app model) | all         | 170 |             0.03 |       -0.12 |         0.17 |             0.09 |
| Router (app model) | stand-alone | 129 |             0    |        0    |         0    |             0    |
| Router (app model) | triathlon   |  41 |             0.12 |       -0.8  |         0.43 |             0.39 |

## 2. How well does the Banister model describe form?

Per athlete with at least 5 race days: each race is forecast from a fit on the earlier races only. Performance = Riegel-equivalent 10 km speed; errors are relative (%). Baselines: running mean of earlier races, previous race. `at_bound` = a fitted time constant sits on its limit (not identified by the data).

| athlete   |   races | method     |   tau1 |   tau2 |   forecasts | at_bound   |   banister_MAPE_% |   running_mean_MAPE_% |   previous_race_MAPE_% |
|:----------|--------:|:-----------|-------:|-------:|------------:|:-----------|------------------:|----------------------:|-----------------------:|
| strava_1  |       5 | fixed_taus |  42    |   7    |           2 | False      |             10.22 |                  9.84 |                   7.3  |
| strava_11 |      12 | full       |  80    |  19    |           9 | True       |             29.24 |                 19.48 |                  21.98 |
| strava_12 |      24 | full       |  20    |   3    |          21 | True       |              8.3  |                  7.53 |                   7.89 |
| strava_13 |      24 | full       |  23.75 |  19    |          21 | True       |              9.8  |                 10.01 |                   6.39 |
| strava_14 |      27 | full       |  80    |  19    |          24 | True       |              5.86 |                  4.88 |                   5.77 |
| strava_15 |      33 | full       |  42.99 |   3    |          30 | True       |             17    |                 14.95 |                  19.37 |
| strava_16 |      15 | full       |  80    |  19    |          12 | True       |              5.52 |                  5.29 |                   5.09 |
| strava_18 |       5 | fixed_taus |  42    |   7    |           2 | False      |              8.77 |                  8.57 |                   6.35 |
| strava_6  |       5 | fixed_taus |  42    |   7    |           2 | False      |             15.03 |                 11    |                  18.76 |
| strava_7  |      14 | full       |  80    |   7.93 |          11 | True       |              6.67 |                  7.02 |                   3.9  |

Pooled over 134 forecasts (10 athletes): Banister **11.21 %**, running mean **9.77 %**, previous race **10.34 %**. 7 of 7 full fits have a time constant at a bound.

## Conclusions

Written by hand from the tables above; the numbers are the script's.

1. **Load features do not improve prediction on this data.** Ridge gets slightly *worse* on
   stand-alone races (+0.46 pp, interval 0.05 to 0.82, so not noise); gradient boosting and the
   triathlon segment show no reliable change (intervals include 0); the app's router is unchanged
   because road races use Riegel, which ignores the features. The load features therefore stay
   **off** (`include_load=False`) in every model, including the app's.
2. **The Banister model does not describe form better than a running mean of earlier races**
   (11.2 % vs 9.8 % pooled; the previous race alone gets 10.3 %). All 7 full fits have a time
   constant stuck at a bound, i.e. the data cannot identify them, as expected with 5 to 33 race
   labels per athlete. The code is correct (it recovers known parameters on synthetic data, see
   `tests/test_banister.py`), so the limit is the data, not the implementation.
3. **Likely reasons, none of them tested here:** race labels are noisy (the manual review in
   `docs/REVIEW_RACES.md` found that about a third of the heuristic "races" were not races, and
   this experiment uses the labels as detected); heart rate is a noisy proxy for load; 14 athletes
   is too few for per-athlete physiology; training/query load differ slightly (training
   drops activities without valid heart rate, prediction imputes it); the first ~6 weeks of every
   history underestimate fitness.
4. **What stays useful:** CTL / ATL / TSB are a sound, descriptive picture of training (the
   *Training load* page of the app) and the building blocks the planner will need. They are not a
   proven predictor of race time here.
5. **Consequences for later stages.** The what-if stage (13) should not promise model-based form
   effects from Banister; plan rules (stage 12) do not depend on it. Revisit load features after
   the race labels are cleaned with the review decisions: if they help then, it is a one-line
   change (`ds.with_load()`).
