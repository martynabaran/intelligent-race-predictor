# Stage 10: model per race mode

170 races (41 triathlon runs), 14 athletes. Metrics on total race time; `MedAPE` = median absolute percentage error. `Router` is the fixed choice used by the app; `AutoRouter` re-selects the model per mode inside every training fold (inner grouped CV over Riegel, Ridge, HistGB, Riegel+correction), so its scores are a nested, unbiased estimate of the routing idea.

| protocol      | model                                 |   MAPE_% all |   MAPE_% stand-alone |   MAPE_% triathlon |   MedAPE_% all |   MedAPE_% stand-alone |   MedAPE_% triathlon |   MAE_time_min all |   MAE_time_min stand-alone |   MAE_time_min triathlon |
|:--------------|:--------------------------------------|-------------:|---------------------:|-------------------:|---------------:|-----------------------:|---------------------:|-------------------:|---------------------------:|-------------------------:|
| chronological | Riegel                                |        11.04 |                 8.38 |              20.52 |           5.42 |                   4.36 |                22.37 |               7.66 |                       8.27 |                     5.48 |
| chronological | Ridge (previous app model)            |         9.79 |                10.02 |               8.94 |           4.86 |                   5.15 |                 2.12 |               7.41 |                       8.99 |                     1.79 |
| chronological | HistGB                                |         7.1  |                 7.45 |               5.85 |           4.56 |                   5.03 |                 3.97 |               4.84 |                       5.86 |                     1.21 |
| chronological | Riegel+correction                     |         9.34 |                 7.67 |              15.3  |           7.48 |                   6.12 |                11.32 |               6.15 |                       6.96 |                     3.26 |
| chronological | Router (road=Riegel, triathlon=Ridge) |         8.51 |                 8.38 |               8.94 |           3.82 |                   4.36 |                 2.12 |               6.85 |                       8.27 |                     1.79 |
| chronological | AutoRouter (nested selection)         |         7.27 |                 7.67 |               5.85 |           5.18 |                   6.12 |                 3.97 |               5.7  |                       6.96 |                     1.21 |
| grouped-CV    | Riegel                                |        10.58 |                 6.62 |              23.03 |           7.32 |                   4.57 |                24.7  |               5.12 |                       4.63 |                     6.63 |
| grouped-CV    | Ridge (previous app model)            |         9.73 |                 8.97 |              12.12 |           9.21 |                   8.71 |                10.65 |               4.77 |                       5.13 |                     3.62 |
| grouped-CV    | HistGB                                |        13.97 |                14.54 |              12.2  |          11.07 |                  11.15 |                10.99 |               7.16 |                       8.24 |                     3.75 |
| grouped-CV    | Riegel+correction                     |         9.74 |                 6.69 |              19.34 |           7.01 |                   5.34 |                19.59 |               4.56 |                       4.43 |                     4.95 |
| grouped-CV    | Router (road=Riegel, triathlon=Ridge) |         7.94 |                 6.62 |              12.12 |           5.61 |                   4.57 |                10.65 |               4.39 |                       4.63 |                     3.62 |
| grouped-CV    | AutoRouter (nested selection)         |         8.15 |                 6.86 |              12.2  |           6.42 |                   4.94 |                10.99 |               4.48 |                       4.71 |                     3.75 |

## What AutoRouter chose (6 outer fits)

- `road`: {'Riegel+correction': 2, 'Riegel': 4}
- `triathlon`: {'HistGB': 6}

## Conclusions

*(Written by hand from the run above; `run_modes.py` regenerates only the tables.)*

1. **Routing by race mode is the best approach for unseen athletes.** The app's router reaches **7.9 %** grouped-CV MAPE
   (median 5.6 %, MAE 4.4 min), vs 9.7 % for Ridge (the previous app model) and 10.6 % for Riegel alone. It takes the
   best of both: Riegel on road races (6.6 %), Ridge on triathlon runs (12.1 %).
2. **The gain is not an artefact of choosing the router after seeing the results.** `AutoRouter` re-selects the
   models inside every training fold and still reaches **8.2 %**. Only ~0.2 pp of the router's advantage is
   selection optimism. Road races went to Riegel in 4/6 fits and to Riegel+correction in 2/6. Triathlon runs went
   to HistGB in all 6, which is within noise of Ridge (12.2 vs 12.1 %).
3. **For athletes already in the training data (chronological protocol), learned models help more.** HistGB alone is
   best (7.1 %) and AutoRouter is close (7.3 %). The app predicts for new users, so the grouped-CV numbers are the
   relevant ones. Once a returning user's races are part of training, a learned road model becomes worth it.
4. **The Riegel+correction hybrid does not beat Riegel on road races of unseen athletes** (6.7 vs 6.6 %). It helps for
   known athletes (7.7 vs 8.4 %). The learned exponent barely improves the marathon (see `RESULTS_BREAKDOWN.md`;
   8 races).
5. **The prediction intervals are now per mode.** On the real data the 10–90 % range of road races is 0.93–1.16 × the
   prediction (it was 0.88–1.15 with Ridge for all races), and 0.85–1.24 × for triathlon runs.
6. **Not done:** a separate trail mode (only 9 races with >200 m of climbing) and a dedicated marathon correction
   (8 marathons). Both need more data, e.g. the Kaggle marathons.
