# Baseline results

170 races, 14 athletes. Metrics are on total race time (pace x distance). `chronological` = thesis protocol (latest 20% of each athlete's races held out); `grouped-CV` = whole athletes held out.

| model   | protocol      |   MAE_time_min |   RMSE_time_min |   MAPE_% |   MAE_pace_s_km |   fit_s |
|:--------|:--------------|---------------:|----------------:|---------:|----------------:|--------:|
| Riegel  | chronological |           7.66 |           18.48 |    11.04 |           32.89 |    0.01 |
| Riegel  | grouped-CV    |           5.24 |           11.03 |    11.03 |           32.72 |    0.01 |
| Ridge   | chronological |           7.43 |           18.95 |     9.89 |           26.82 |    0.01 |
| Ridge   | grouped-CV    |           4.73 |            8.09 |     9.69 |           29.77 |    0.01 |
| HistGB  | chronological |           4.72 |           10.93 |     7.25 |           19.02 |    0.99 |
| HistGB  | grouped-CV    |           6.86 |           11.94 |    13.59 |           41.97 |    0.99 |
| GRU     | chronological |           9.34 |           28.55 |    10.29 |           29.07 |  616.32 |
| GRU     | grouped-CV    |           7.18 |           12.09 |    13.87 |           45.37 |  616.32 |
