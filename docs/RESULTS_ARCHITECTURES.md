# Sequence architectures: GRU vs TCN vs Transformer

170 races, 14 athletes; identical hybrid layout (encoder + static branch), mean over 3 seeds (`MAPE_std` = std of MAPE across seeds). Metrics on total race time. With so few races, differences smaller than `MAPE_std` should not be over-interpreted.

| model       | protocol      |   MAE_time_min |   RMSE_time_min |   MAPE_% |   MAE_pace_s_km |   fit_s |   MAPE_std |
|:------------|:--------------|---------------:|----------------:|---------:|----------------:|--------:|-----------:|
| GRU         | chronological |           7.13 |           17.4  |     9.74 |           26.29 |  498.11 |       0.44 |
| GRU         | grouped-CV    |           7.3  |           12.47 |    15.04 |           47.08 |  498.11 |       2.5  |
| TCN         | chronological |           7.55 |           15.01 |    12.3  |           34.02 |   95.23 |       1.5  |
| TCN         | grouped-CV    |           8.04 |           13.45 |    17.03 |           53.42 |   95.23 |       1.9  |
| Transformer | chronological |           9.14 |           23.51 |    12.63 |           35.15 |  120.9  |       2.4  |
| Transformer | grouped-CV    |           8.6  |           15.49 |    17.7  |           53.42 |  120.9  |       3.04 |
