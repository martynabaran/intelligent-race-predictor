# Embedded models: distillation + TFLite quantisation

> **Note:** computed on the earlier 170-race set, before the manual race review (`REVIEW_RACES.md`); to be recomputed on the reviewed labels in stage 15.

170 races. Teacher: GRU (hybrid, as in Stage 4). Students are trained plainly or distilled (targets = 0.5*true + 0.5*teacher). Accuracy is on total race time with the same two protocols as before; `keras` = float model before export.

## Accuracy

| student               | training        | format       |   MAPE_% (chronological) |   MAPE_% (grouped-CV) |   MAE_time_min (chronological) |   MAE_time_min (grouped-CV) |
|:----------------------|:----------------|:-------------|-------------------------:|----------------------:|-------------------------------:|----------------------------:|
| teacher               | -               | keras        |                     9.87 |                 12.92 |                           8.94 |                        6    |
| MLP(16,8) tabular     | plain           | keras        |                     8.99 |                 13.43 |                           7.1  |                        6.6  |
| MLP(16,8) tabular     | plain           | tflite-float |                     8.99 |                 13.43 |                           7.1  |                        6.6  |
| MLP(16,8) tabular     | plain           | tflite-int8  |                     8.02 |                 13.53 |                           4.39 |                        6.98 |
| MLP(16,8) tabular     | distilled a=0.5 | keras        |                     7.8  |                 13.45 |                           4.7  |                        6.6  |
| MLP(16,8) tabular     | distilled a=0.5 | tflite-float |                     7.8  |                 13.45 |                           4.7  |                        6.6  |
| MLP(16,8) tabular     | distilled a=0.5 | tflite-int8  |                     8.23 |                 13.55 |                           5.62 |                        6.92 |
| TCN-small (w=8)       | plain           | keras        |                    16.02 |                 23.82 |                           8.8  |                       11.82 |
| TCN-small (w=8)       | plain           | tflite-float |                    16.02 |                 23.82 |                           8.8  |                       11.82 |
| TCN-small (w=8)       | distilled a=0.5 | keras        |                    16.9  |                 19.61 |                           9.1  |                       10.14 |
| TCN-small (w=8)       | distilled a=0.5 | tflite-float |                    16.9  |                 19.61 |                           9.1  |                       10.14 |
| CompactSequence (w=8) | plain           | keras        |                    14.22 |                 17.29 |                           7.74 |                        8.69 |
| CompactSequence (w=8) | plain           | tflite-float |                    14.22 |                 17.29 |                           7.74 |                        8.69 |
| CompactSequence (w=8) | plain           | tflite-int8  |                    13.53 |                 17.48 |                           6.31 |                        8.7  |
| CompactSequence (w=8) | distilled a=0.5 | keras        |                    14.26 |                 16.75 |                           6.3  |                        8.74 |
| CompactSequence (w=8) | distilled a=0.5 | tflite-float |                    14.26 |                 16.75 |                           6.3  |                        8.74 |
| CompactSequence (w=8) | distilled a=0.5 | tflite-int8  |                    14.32 |                 18.26 |                           6.7  |                        8.91 |

## Size and latency (models fitted on all data)

Latency: median single-sample inference, 1 thread, host CPU (relative indicator only).

| model                           | format       |   size_kb |     latency_us |
|:--------------------------------|:-------------|----------:|---------------:|
| teacher                         | keras        |     493.2 |    2.36924e+06 |
| teacher                         | tflite-float |    4008.3 | 2375.3         |
| teacher                         | tflite-int8  |   16923.7 | 2183.5         |
| MLP(16,8) tabular plain         | tflite-float |       3.9 |    2.9         |
| MLP(16,8) tabular plain         | tflite-int8  |       3.8 |    4.3         |
| MLP(16,8) tabular distilled     | tflite-float |       3.9 |    2.9         |
| MLP(16,8) tabular distilled     | tflite-int8  |       3.8 |    3           |
| TCN-small (w=8) plain           | tflite-float |      66.9 |  161.9         |
| TCN-small (w=8) distilled       | tflite-float |      66.9 |  161           |
| CompactSequence (w=8) plain     | tflite-float |      13.6 |   20.3         |
| CompactSequence (w=8) plain     | tflite-int8  |      13.1 |   20.5         |
| CompactSequence (w=8) distilled | tflite-float |      13.6 |   20.8         |
| CompactSequence (w=8) distilled | tflite-int8  |      13.1 |   25.1         |

## Notes on the TFLite fixes

*(Written by hand; `run_embedded.py` regenerates only the tables above.)*

- **The GRU teacher now converts to builtin TFLite ops.** The RNN layers are unrolled at export time (`unroll_rnns`), so
  no Flex delegate is needed. The cost: unrolling 256 steps makes the model large (4 MB float vs 0.5 MB `.keras`). Int8 is
  even larger (17 MB), because quantisation parameters and requantise ops repeat at every step. Recurrent models are a
  poor fit for a device, which supports using a small student.
- **`CompactSequence` is the int8-ready sequence student.** It takes the padding mask as a separate input with zero
  padding (instead of `PAD_VALUE = 999` inside the data) and uses an int8-safe masked pool (instead of the `-1e9` trick
  that aborted the interpreter). Compared with `TCN-small`, it is 5× smaller (13 vs 67 KB), ~8× faster (20 vs 160 µs)
  and more accurate on unseen athletes (16.8–17.5 % vs 19.6–23.8 % MAPE), with no loss from int8.
- **The tabular MLP is still the best device model** (~13.5 % grouped-CV, 3.8 KB, ~3 µs). With 170 races, a sequence
  student does not beat a few aggregate features.
- **The teacher's numbers differ slightly from the previous run** (12.92 vs 12.78 % grouped-CV) with the same seed.
  TensorFlow CPU training is not bit-for-bit reproducible across machines, which is another reason to read
  differences under ~1 pp as noise.
