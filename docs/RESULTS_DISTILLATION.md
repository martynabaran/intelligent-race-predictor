# Knowledge distillation and TFLite

170 races, 14 athletes; teacher: `tcn`, mean over 3 seed(s). The teacher is fitted once per split and shared by the distilled student and its TFLite exports. Metrics on total race time; `MAPE_std` = std of MAPE across seeds - differences smaller than it should not be over-interpreted.

## Accuracy

| model                        | protocol      |   MAE_time_min |   RMSE_time_min |   MAPE_% |   MAE_pace_s_km |   MAPE_std |
|:-----------------------------|:--------------|---------------:|----------------:|---------:|----------------:|-----------:|
| Teacher (TCN)                | chronological |           9.55 |           20.74 |    13.28 |           37.98 |       4.5  |
| Compact (no distillation)    | chronological |           8.51 |           17.45 |    13.39 |           37.37 |       1.63 |
| Distilled                    | chronological |           8.14 |           14.78 |    13.42 |           35.8  |       2.53 |
| Teacher (TCN) TFLite-float32 | chronological |           9.55 |           20.74 |    13.28 |           37.98 |       4.5  |
| Distilled TFLite-float32     | chronological |           8.14 |           14.78 |    13.42 |           35.8  |       2.53 |
| Distilled TFLite-dynamic     | chronological |           8.14 |           14.78 |    13.42 |           35.8  |       2.53 |
| Distilled TFLite-int8        | chronological |           8.2  |           15.17 |    13.22 |           35.33 |       3    |
| Teacher (TCN)                | grouped-CV    |           8.04 |           13.19 |    16.96 |           52.15 |       2.73 |
| Compact (no distillation)    | grouped-CV    |           8.79 |           15.64 |    18.19 |           55.26 |       0.04 |
| Distilled                    | grouped-CV    |           8.69 |           16.14 |    17.89 |           55.01 |       1.7  |
| Teacher (TCN) TFLite-float32 | grouped-CV    |           8.04 |           13.19 |    16.96 |           52.15 |       2.73 |
| Distilled TFLite-float32     | grouped-CV    |           8.69 |           16.14 |    17.89 |           55.01 |       1.7  |
| Distilled TFLite-dynamic     | grouped-CV    |           8.69 |           16.14 |    17.89 |           55.01 |       1.7  |
| Distilled TFLite-int8        | grouped-CV    |           8.11 |           13.4  |    16.2  |           50.66 |       0.31 |

## Footprint

Single-race CPU latency (median of 100 runs, including preprocessing); Keras size is float32 weights, TFLite size is the flatbuffer.

| model                        |   params |   size_kb |   latency_ms |
|:-----------------------------|---------:|----------:|-------------:|
| Teacher (TCN)                |    49281 |     192.5 |       44.644 |
| Compact (no distillation)    |     1177 |       4.6 |       46.464 |
| Distilled                    |     1177 |       4.6 |       46.299 |
| Teacher (TCN) TFLite-float32 |    49281 |     240.9 |        1.226 |
| Distilled TFLite-float32     |     1177 |      13.5 |        0.606 |
| Distilled TFLite-dynamic     |     1177 |      13.5 |        0.606 |
| Distilled TFLite-int8        |     1177 |      12.9 |        0.601 |

## Conclusions

*(Written by hand from the run above; `run_distillation.py` regenerates only the tables.)*

1. **The student matches the teacher within seed noise, with 42× fewer parameters** (1,177 vs 49,281). MAPE is 13.4 vs
   13.3 % (chronological) and 17.9 vs 17.0 % (grouped-CV), with `MAPE_std` of 1.7–4.5. Under the chronological protocol
   the student even has lower RMSE (14.8 vs 20.7 min), so it makes fewer very large errors.
2. **Distillation vs. training the same student from scratch: no clear gain at this data size.** Distillation gives slightly
   lower MAE/RMSE in the chronological protocol and equal results in grouped-CV, all within noise. Most of the benefit comes
   from the small architecture itself, which over-fits less on 170 races.
3. **Dynamic-range quantisation is identical to float32 here.** The converter only quantises weight tensors with at least
   1024 elements, and every student tensor is smaller, so nothing is quantised.
4. **Full int8 costs no accuracy** (within noise of float32 in both protocols). The file shrinks only from 13.5 to 12.9 KB,
   because at 1.2 k parameters the flatbuffer overhead dominates. The weights themselves go from ~4.6 KB to ~1.2 KB.
5. **Latency.** TFLite answers a single race in ~0.6 ms vs ~46 ms for Keras `predict()`; most of that gap is Keras per-call
   overhead. On TFLite the student is ~2× faster than the teacher (0.6 vs 1.2 ms). Both are sub-millisecond on a laptop
   CPU; the size and int8 support matter on a microcontroller-class device (e.g. a sports watch).
6. **Context.** In grouped-CV every sequence model here (~17 % MAPE) is still worse than Ridge from Stage 2 (9.7 %).
   With 14 athletes, the main result of this stage is the deployment path (small, int8, sub-ms), not a more accurate
   model. The Stage 3 Kaggle pre-training run may change the accuracy picture.
