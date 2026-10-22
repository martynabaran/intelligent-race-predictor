# Baseline results

129 races, 12 athletes. Metrics are on total race time (pace x distance). `chronological` = thesis protocol (latest 20% of each athlete's races held out); `grouped-CV` = whole athletes held out.

| model   | protocol      |   MAE_time_min |   RMSE_time_min |   MAPE_% |   MedAPE_% |   MAE_pace_s_km |   fit_s |
|:--------|:--------------|---------------:|----------------:|---------:|-----------:|----------------:|--------:|
| Riegel  | chronological |           8.79 |           21.07 |     9.46 |       5.42 |           29.22 |    0.02 |
| Riegel  | grouped-CV    |           4.63 |           11.51 |     6.62 |       4.57 |           20.94 |    0.02 |
| Ridge   | chronological |           5.19 |           11.54 |     7.8  |       6.24 |           21.47 |    0.01 |
| Ridge   | grouped-CV    |           6.13 |           12.1  |    10.66 |       8.45 |           31.1  |    0.01 |
| HistGB  | chronological |          12.12 |           27.59 |    11.53 |       6.64 |           34.93 |    1.39 |
| HistGB  | grouped-CV    |           8.35 |           14.07 |    14.84 |      14.4  |           48.65 |    1.39 |
| GRU     | chronological |           6.29 |           10.56 |     8.43 |       8.28 |           23.21 | 1907.2  |
| GRU     | grouped-CV    |           9.2  |           16.52 |    16.78 |      15.2  |           52.55 | 1907.2  |
