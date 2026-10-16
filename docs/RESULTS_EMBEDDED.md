# Embedded models: distillation + TFLite quantisation

170 races. Teacher: GRU (hybrid, as in Stage 4). Students are trained plainly or distilled (targets = 0.5*true + 0.5*teacher). Accuracy is on total race time with the same two protocols as before; `keras` = float model before export.

## Accuracy

| student           | training        | format       |   MAPE_% (chronological) |   MAPE_% (grouped-CV) |   MAE_time_min (chronological) |   MAE_time_min (grouped-CV) |
|:------------------|:----------------|:-------------|-------------------------:|----------------------:|-------------------------------:|----------------------------:|
| teacher           | -               | keras        |                     9.4  |                 12.78 |                           7.53 |                        6.3  |
| MLP(16,8) tabular | plain           | keras        |                     8.9  |                 13.48 |                           7.19 |                        6.55 |
| MLP(16,8) tabular | plain           | tflite-float |                     8.9  |                 13.48 |                           7.19 |                        6.55 |
| MLP(16,8) tabular | plain           | tflite-int8  |                     8.31 |                 13.44 |                           5.17 |                        6.87 |
| MLP(16,8) tabular | distilled a=0.5 | keras        |                     9.06 |                 12.79 |                           7.42 |                        6.33 |
| MLP(16,8) tabular | distilled a=0.5 | tflite-float |                     9.06 |                 12.79 |                           7.42 |                        6.33 |
| MLP(16,8) tabular | distilled a=0.5 | tflite-int8  |                     8.22 |                 12.91 |                           5.08 |                        6.65 |
| TCN-small (w=8)   | plain           | keras        |                    15.72 |                 24.03 |                           8.26 |                       10.29 |
| TCN-small (w=8)   | plain           | tflite-float |                    15.72 |                 24.03 |                           8.26 |                       10.29 |
| TCN-small (w=8)   | distilled a=0.5 | keras        |                    15.22 |                 21.13 |                           8.13 |                        9.48 |
| TCN-small (w=8)   | distilled a=0.5 | tflite-float |                    15.22 |                 21.13 |                           8.13 |                        9.48 |

## Size and latency (models fitted on all data)

Latency: median single-sample inference, 1 thread, host CPU (relative indicator only).

| model                       | format       |   size_kb |    latency_us | note                            |
|:----------------------------|:-------------|----------:|--------------:|:--------------------------------|
| teacher                     | keras        |     492.1 |   1.06997e+06 | nan                             |
| teacher                     | tflite-float |     nan   | nan           | conversion failed: RuntimeError |
| MLP(16,8) tabular plain     | tflite-float |       3.9 |   1.6         | nan                             |
| MLP(16,8) tabular plain     | tflite-int8  |       3.7 |   1.6         | nan                             |
| MLP(16,8) tabular distilled | tflite-float |       3.9 |   1.6         | nan                             |
| MLP(16,8) tabular distilled | tflite-int8  |       3.7 |   1.6         | nan                             |
| TCN-small (w=8) plain       | tflite-float |      71.3 |  99.4         | nan                             |
| TCN-small (w=8) distilled   | tflite-float |      71.3 |  99           | nan                             |
