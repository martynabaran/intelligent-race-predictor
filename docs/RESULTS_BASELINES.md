# Baseline results

125 races, 13 athletes. Metrics are on total race time (pace x distance). `chronological` = thesis protocol (latest 20% of each athlete's races held out); `grouped-CV` = whole athletes held out.

| model   | protocol      |   MAE_time_min |   RMSE_time_min |   MAPE_% |   MedAPE_% |   MAE_pace_s_km |   fit_s |
|:--------|:--------------|---------------:|----------------:|---------:|-----------:|----------------:|--------:|
| Riegel  | chronological |           7.74 |           20.72 |     7.49 |       4.66 |           22.37 |    0.01 |
| Riegel  | grouped-CV    |           4.66 |           11.86 |     7.06 |       4.77 |           20.57 |    0.01 |
| Ridge   | chronological |           5.46 |           12.14 |     8.08 |       6.55 |           20.61 |    0.01 |
| Ridge   | grouped-CV    |           7.01 |           11.78 |    14.01 |      13.73 |           41.4  |    0.01 |
| HistGB  | chronological |           7.72 |           17.6  |     8.66 |       4.37 |           23.94 |    1.33 |
| HistGB  | grouped-CV    |           7.93 |           14    |    14.91 |      12.09 |           46.03 |    1.33 |
