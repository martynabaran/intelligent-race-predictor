# Error breakdown and data cleaning

`thesis` = the thesis cleaning chain; `fast-run filter` = plus `RemoveImplausiblyFastRuns` (same races as `thesis`, so directly comparable); `+ HR-spike filter` = plus `ValidateMaxHeartRate` (now default), which lowers inflated HRmax and therefore detects more races; `+ reviewed labels` = plus the hand-labelled races of `race_labels.csv` (the current default of the app's training data). The last two are on different race sets than the first two. Metrics on total race time; `MedAPE` = median absolute percentage error. `stand-alone` excludes triathlon runs - the app only predicts stand-alone races.

## Datasets

|                   |   races |   athletes |   triathlon runs |   stand-alone |
|:------------------|--------:|-----------:|-----------------:|--------------:|
| thesis            |     170 |         14 |               41 |           129 |
| fast-run filter   |     170 |         14 |               41 |           129 |
| + HR-spike filter |     203 |         14 |               45 |           158 |
| + reviewed labels |     143 |         13 |               18 |           125 |

## MAPE / median APE by segment

| model   | protocol      | pipeline          |   MAPE_% all |   MAPE_% stand-alone |   MAPE_% triathlon |   MedAPE_% all |   MedAPE_% stand-alone |   MedAPE_% triathlon |
|:--------|:--------------|:------------------|-------------:|---------------------:|-------------------:|---------------:|-----------------------:|---------------------:|
| Riegel  | chronological | thesis            |        11.04 |                 8.38 |              20.52 |           5.42 |                   4.36 |                22.37 |
| Riegel  | chronological | fast-run filter   |        11.04 |                 8.38 |              20.52 |           5.42 |                   4.36 |                22.37 |
| Riegel  | chronological | + HR-spike filter |        12.03 |                 8.17 |              21.86 |           8.79 |                   4.6  |                23.39 |
| Riegel  | chronological | + reviewed labels |         8.81 |                 7.06 |              16.49 |           6    |                   4.6  |                16.94 |
| Riegel  | grouped-CV    | thesis            |        11.03 |                 6.86 |              24.13 |           7.4  |                   4.58 |                26.49 |
| Riegel  | grouped-CV    | fast-run filter   |        10.58 |                 6.62 |              23.03 |           7.32 |                   4.57 |                24.7  |
| Riegel  | grouped-CV    | + HR-spike filter |        10.57 |                 7.27 |              22.16 |           7.62 |                   4.86 |                23.52 |
| Riegel  | grouped-CV    | + reviewed labels |         8.15 |                 7.06 |              15.73 |           5.55 |                   4.77 |                15.58 |
| Ridge   | chronological | thesis            |         9.89 |                10.23 |               8.67 |           4.92 |                   5.32 |                 1.25 |
| Ridge   | chronological | fast-run filter   |         9.79 |                10.02 |               8.94 |           4.86 |                   5.15 |                 2.12 |
| Ridge   | chronological | + HR-spike filter |         8.97 |                 8.91 |               9.14 |           6.17 |                   6.57 |                 4.9  |
| Ridge   | chronological | + reviewed labels |         8.55 |                 8.86 |               7.21 |           7.28 |                   7.15 |                 7.41 |
| Ridge   | grouped-CV    | thesis            |         9.69 |                 8.81 |              12.48 |           8.93 |                   8.48 |                10.65 |
| Ridge   | grouped-CV    | fast-run filter   |         9.73 |                 8.97 |              12.12 |           9.21 |                   8.71 |                10.65 |
| Ridge   | grouped-CV    | + HR-spike filter |        10.56 |                 9.54 |              14.14 |           9.5  |                   8.79 |                12.92 |
| Ridge   | grouped-CV    | + reviewed labels |        11.6  |                11.19 |              14.46 |           9.84 |                   9.49 |                11.57 |
| HistGB  | chronological | thesis            |         7.25 |                 7.56 |               6.15 |           3.99 |                   4.06 |                 3.68 |
| HistGB  | chronological | fast-run filter   |         7.1  |                 7.45 |               5.85 |           4.56 |                   5.03 |                 3.97 |
| HistGB  | chronological | + HR-spike filter |         7.74 |                 7.56 |               8.18 |           5.75 |                   4.76 |                 7.04 |
| HistGB  | chronological | + reviewed labels |         7.33 |                 7.45 |               6.81 |           4.97 |                   4.81 |                 9.54 |
| HistGB  | grouped-CV    | thesis            |        13.59 |                14    |              12.28 |          10.66 |                  10.38 |                10.99 |
| HistGB  | grouped-CV    | fast-run filter   |        13.97 |                14.54 |              12.2  |          11.07 |                  11.15 |                10.99 |
| HistGB  | grouped-CV    | + HR-spike filter |        13.01 |                13.84 |              10.1  |          10.02 |                  10.61 |                 8.11 |
| HistGB  | grouped-CV    | + reviewed labels |        13.29 |                14.01 |               8.25 |          12.05 |                  12.48 |                 6.59 |

## MAPE per distance (stand-alone races, fast-run filter = default pipeline, grouped-CV)

| segment                |   n |   Riegel |   Ridge |   HistGB |
|:-----------------------|----:|---------:|--------:|---------:|
| 5k (stand-alone)       |  25 |     4.51 |    7.77 |    12.38 |
| marathon (stand-alone) |   8 |    13.17 |   10.36 |    17.14 |
| other (stand-alone)    |  50 |     8.04 |    8.88 |    14.31 |
| half (stand-alone)     |   9 |     4.11 |   10.09 |    14.24 |
| 10k (stand-alone)      |  37 |     5.31 |    9.33 |    15.81 |

## Conclusions

*(Written by hand from the run above; `run_breakdown.py` regenerates only the tables.)*

1. **Triathlon runs cause most of the error.** A quarter of the races (41/170) are the run leg of a triathlon. Riegel's
   grouped-CV MAPE is 6.6 % on stand-alone races and 23 % on triathlon runs. The pooled ~11 % mixes the two.
2. **For stand-alone races, the ones the app predicts, Riegel is the best model on unseen athletes.** Grouped-CV MAPE is
   6.6 % (median 4.6 %) for Riegel, vs 9.0 % for Ridge and 14.5 % for HistGB. Ridge looked best overall only because
   `is_triathlon` lets it learn the triathlon slowdown. When the athlete's earlier races are in the training data
   (chronological protocol), HistGB is better (7.5 % vs 8.4 %), so personal history does help returning users.
3. **The mean overstates the typical error.** The median APE of Riegel on stand-alone races is ~4.6 %, about 2 minutes on
   a 45-minute 10 k. The mean is pulled up by a few large misses.
4. **Per distance, Riegel is 4–5 % on 5 k, 10 k and half-marathon but 13 % on the marathon**, where Ridge does better
   (10.4 %). This matches the known weakness of Riegel's fixed exponent: amateurs slow down more over a marathon. There are
   only 8 marathons, though. 50 stand-alone "races" are not within ±10 % of a standard distance, which is worth a look at
   the race detector.
5. **Fast-run filter (now on by default).** It removes exactly one activity: a 12 km "run" at 2:38/km with average HR 131,
   a ride logged as a run. The race set is unchanged, so the comparison with `thesis` is like-for-like. Riegel's
   grouped-CV MAPE goes 11.0 → 10.6 % (triathlon runs 24.1 → 23.0 %). The ML models change within noise.
6. **HR-spike filter (now default).** 16 `max_hr` readings above 220 bpm had inflated HRmax to 220–253 bpm
   for 6 athletes. Removing them lowers the HR threshold of the race detector, which then finds 33 more
   races. Because the race set changes, these metrics are not comparable to the first two pipelines. The
   manual review found all 33 of these races to be real, so the filter is on by default.
7. **Runs after a ride.** The review lists (`REVIEW_RACES.md`) covered the 39 post-ride candidates detected only
   because they follow a ride and have a standard distance: 27 were brick training sessions, 12 real races.
   The detector cannot tell them apart, so runs after a ride are not road races to predict
   (`DropTriathlonRaces`, default), and the app predicts road races only (see `RESULTS_MODES.md`). This
   report keeps them (`include_triathlon_races=True`) because it documents the analysis.
8. **Reviewed labels (`+ reviewed labels`).** Applying the manual decisions removes 33 stand-alone "races" that
   were not races (32 of them were flagged by a high heart rate alone, a signal that was right only 14 times out
   of 46) and keeps the 33 from the spike filter: 125 stand-alone races from 13 athletes (143 with the 18 post-ride
   races kept as training data). On the stand-alone races Riegel's grouped-CV MAPE is 7.1 % (median 4.8 %) and
   Ridge's 11.2 % here (14.0 % in `RESULTS_MODES.md`, which trains without the post-ride races). On a race set with
   correct labels the learned models' disadvantage on unseen athletes is larger than the earlier tables suggested.
