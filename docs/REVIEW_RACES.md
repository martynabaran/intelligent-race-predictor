# Przegląd wykrytych startów

Wygenerowane przez `experiments/review_races.py`. W pracy startów nie ma w danych jako etykiet - wykrywa je heurystyka `HeuristicRaceDetector` (punkty: wysokie tętno +2, skok prędkości vs średnia z roku +2, >40 km +2, bieg po rowerze +1, standardowy dystans +1; start = co najmniej 2 punkty). Proszę o wpisanie w kolumnie *decyzja* `tak` / `nie` / `?` (znasz te dane najlepiej).

Dane modelu: **170 startów** (41 po rowerze).

- **A.** Starty wykryte tylko dlatego, że są po rowerze i mają standardowy dystans: **39** z 41 startów po rowerze.
- **B.** Starty dochodzące po włączeniu filtra skoków tętna: **33** (znikające: 0).
- **C.** Zwykłe starty o niestandardowym dystansie: **50**.

## A. Tylko „po rowerze” + standardowy dystans (może trening typu brick?)

Bez wysokiego tętna i bez skoku prędkości. Jeśli to w większości treningi, a nie zawody, trzeba podnieść próg albo wymagać wysokiego tętna dla biegów po rowerze.

| athlete   | date       |    km | pace /km   |   avg HR |   HRmax |   % HRmax | after ride   | reasons                       | decyzja (start?)   |
|:----------|:-----------|------:|:-----------|---------:|--------:|----------:|:-------------|:------------------------------|:-------------------|
| garmin_2  | 2025-05-11 |  4.91 | 5:23       |      135 |     195 |        69 | tak          | after_ride, standard_distance |                    |
| strava_11 | 2021-03-29 | 10.52 | 8:08       |      144 |     230 |        63 | tak          | after_ride, standard_distance |                    |
| strava_11 | 2022-03-12 |  4.95 | 4:40       |      177 |     230 |        77 | tak          | after_ride, standard_distance |                    |
| strava_11 | 2022-07-15 |  9.56 | 5:46       |      147 |     230 |        64 | tak          | after_ride, standard_distance |                    |
| strava_11 | 2023-06-03 |  4.52 | 3:53       |      182 |     230 |        79 | tak          | after_ride, standard_distance |                    |
| strava_11 | 2024-08-07 |  4.85 | 4:07       |      184 |     230 |        80 | tak          | after_ride, standard_distance |                    |
| strava_13 | 2024-01-26 | 10.02 | 4:55       |      162 |     190 |        85 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2021-07-19 |  5.05 | 4:44       |      153 |     244 |        63 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2021-07-22 |  4.92 | 4:58       |      147 |     244 |        60 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2022-09-11 |  4.74 | 4:35       |      152 |     244 |        62 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2023-08-11 |  5    | 3:53       |      177 |     244 |        73 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2023-08-12 |  5.03 | 4:24       |      159 |     244 |        65 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2024-03-28 |  5.11 | 4:11       |      156 |     244 |        64 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2024-05-03 |  4.87 | 4:06       |      161 |     244 |        66 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2024-08-21 |  4.54 | 4:36       |      145 |     244 |        59 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-02-18 |  4.5  | 5:06       |      144 |     244 |        59 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-02-23 |  5.03 | 4:48       |      142 |     244 |        58 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-03-10 |  5.03 | 4:29       |      155 |     244 |        64 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-04-16 |  4.63 | 4:20       |      171 |     244 |        70 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-04-23 |  5.01 | 4:37       |      144 |     244 |        59 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-05-08 |  5.01 | 4:35       |      142 |     244 |        58 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-05-11 |  9.72 | 4:54       |      143 |     244 |        59 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-05-14 |  5.19 | 4:33       |      144 |     244 |        59 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-06-04 |  5.01 | 4:36       |      147 |     244 |        60 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-06-12 |  5.03 | 5:03       |      134 |     244 |        55 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-06-21 |  5.02 | 3:55       |      162 |     244 |        66 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-07-19 |  5    | 4:14       |      158 |     244 |        65 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-07-19 |  5.03 | 4:14       |      161 |     244 |        66 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-07-23 |  5.01 | 4:49       |      134 |     244 |        55 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-07-26 | 10.02 | 4:14       |      156 |     244 |        64 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-07-31 |  5    | 4:37       |      140 |     244 |        57 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-08-02 |  5.01 | 4:33       |      143 |     244 |        59 | tak          | after_ride, standard_distance |                    |
| strava_15 | 2025-09-11 |  4.93 | 4:32       |      143 |     244 |        59 | tak          | after_ride, standard_distance |                    |
| strava_18 | 2023-05-13 |  4.55 | 3:52       |      175 |     222 |        79 | tak          | after_ride, standard_distance |                    |
| strava_18 | 2023-05-24 |  5.02 | 3:56       |      158 |     222 |        71 | tak          | after_ride, standard_distance |                    |
| strava_18 | 2023-07-27 |  4.85 | 4:04       |      133 |     222 |        60 | tak          | after_ride, standard_distance |                    |
| strava_18 | 2024-03-27 | 10.03 | 4:38       |      152 |     222 |        68 | tak          | after_ride, standard_distance |                    |
| strava_18 | 2025-07-03 | 10.82 | 4:29       |      153 |     222 |        69 | tak          | after_ride, standard_distance |                    |
| strava_6  | 2022-10-05 |  5.19 | 4:31       |      176 |     207 |        85 | tak          | after_ride, standard_distance |                    |

## B. Dochodzą po filtrze skoków tętna (realne HRmax)

`HRmax before` = HRmax zawyżone przez skoki czujnika; `HRmax` = po filtrze. Niższe HRmax obniża próg tętna, więc te biegi zaczynają spełniać warunek wysokiego tętna.

| athlete   | date       |    km | pace /km   |   avg HR |   HRmax before |   HRmax |   % HRmax | after ride   | reasons                    | decyzja (start?)   |
|:----------|:-----------|------:|:-----------|---------:|---------------:|--------:|----------:|:-------------|:---------------------------|:-------------------|
| strava_11 | 2021-01-24 |  3.44 | 4:38       |      194 |            230 |     218 |        89 |              | high_hr                    |                    |
| strava_11 | 2021-02-07 |  3.45 | 4:39       |      194 |            230 |     218 |        89 |              | high_hr                    |                    |
| strava_11 | 2021-02-21 |  3.37 | 4:45       |      195 |            230 |     218 |        89 |              | high_hr                    |                    |
| strava_11 | 2021-03-06 |  3.36 | 4:39       |      191 |            230 |     218 |        88 |              | high_hr                    |                    |
| strava_11 | 2021-07-24 |  3    | 3:40       |      189 |            230 |     218 |        87 | tak          | high_hr, after_ride        |                    |
| strava_11 | 2021-09-04 |  4.95 | 4:20       |      194 |            230 |     218 |        89 |              | high_hr, standard_distance |                    |
| strava_11 | 2021-09-19 |  5.41 | 4:13       |      190 |            230 |     218 |        87 |              | high_hr, standard_distance |                    |
| strava_11 | 2021-10-03 |  4.07 | 3:53       |      194 |            230 |     218 |        89 |              | high_hr                    |                    |
| strava_11 | 2021-12-05 |  5.61 | 5:04       |      191 |            230 |     218 |        88 |              | high_hr                    |                    |
| strava_11 | 2021-12-31 |  5.03 | 4:03       |      196 |            230 |     218 |        90 |              | high_hr, standard_distance |                    |
| strava_11 | 2022-02-06 |  5.6  | 4:59       |      191 |            230 |     218 |        88 |              | high_hr                    |                    |
| strava_11 | 2022-09-11 |  4.24 | 4:15       |      196 |            230 |     218 |        90 |              | high_hr                    |                    |
| strava_11 | 2023-05-13 |  4.65 | 3:47       |      191 |            230 |     218 |        88 |              | high_hr, standard_distance |                    |
| strava_11 | 2023-11-25 |  3.14 | 3:39       |      190 |            230 |     218 |        87 |              | high_hr                    |                    |
| strava_11 | 2024-04-14 |  5.02 | 3:42       |      194 |            230 |     218 |        89 |              | high_hr, standard_distance |                    |
| strava_11 | 2024-05-19 |  6.13 | 4:41       |      189 |            230 |     218 |        87 | tak          | high_hr, after_ride        |                    |
| strava_11 | 2024-06-23 |  5.05 | 3:53       |      191 |            230 |     218 |        88 |              | high_hr, standard_distance |                    |
| strava_11 | 2025-05-11 |  5.06 | 3:51       |      197 |            230 |     218 |        90 |              | high_hr, standard_distance |                    |
| strava_15 | 2021-07-13 |  3.01 | 3:42       |      185 |            244 |     205 |        90 | tak          | high_hr, after_ride        |                    |
| strava_15 | 2022-02-20 |  7.9  | 3:41       |      180 |            244 |     205 |        88 |              | high_hr                    |                    |
| strava_15 | 2022-03-13 |  7.83 | 3:34       |      182 |            244 |     205 |        89 |              | high_hr                    |                    |
| strava_15 | 2023-01-15 |  7.95 | 3:35       |      182 |            244 |     205 |        89 |              | high_hr                    |                    |
| strava_15 | 2023-06-18 |  6.17 | 3:38       |      178 |            244 |     205 |        87 | tak          | high_hr, after_ride        |                    |
| strava_15 | 2024-01-21 |  8.02 | 3:58       |      179 |            244 |     205 |        87 |              | high_hr                    |                    |
| strava_15 | 2024-10-05 |  9.91 | 3:22       |      179 |            244 |     205 |        87 |              | high_hr, standard_distance |                    |
| strava_15 | 2024-10-12 | 20.31 | 4:31       |      179 |            244 |     205 |        87 |              | high_hr, standard_distance |                    |
| strava_18 | 2022-10-16 | 21.23 | 4:23       |      184 |            222 |     213 |        86 |              | high_hr, standard_distance |                    |
| strava_18 | 2022-10-31 |  5.5  | 3:56       |      185 |            222 |     213 |        87 |              | high_hr, standard_distance |                    |
| strava_18 | 2023-03-19 | 21.28 | 5:00       |      184 |            222 |     213 |        86 |              | high_hr, standard_distance |                    |
| strava_18 | 2023-06-11 | 10.98 | 4:01       |      186 |            222 |     213 |        87 |              | high_hr, standard_distance |                    |
| strava_18 | 2023-09-10 | 10.56 | 4:19       |      187 |            222 |     213 |        88 |              | high_hr, standard_distance |                    |
| strava_18 | 2024-04-13 | 10.09 | 3:44       |      185 |            222 |     213 |        87 |              | high_hr, standard_distance |                    |
| strava_8  | 2025-04-20 |  5.26 | 3:46       |      183 |            240 |     209 |        88 |              | high_hr, standard_distance |                    |

## C. Zwykłe starty o niestandardowym dystansie

Np. biegi na 7 km, 15 km, trail lub niedokładny GPS. Czy to zawody?

| athlete   | date       |    km | pace /km   |   avg HR |   HRmax |   % HRmax | after ride   | reasons              | decyzja (start?)   |
|:----------|:-----------|------:|:-----------|---------:|--------:|----------:|:-------------|:---------------------|:-------------------|
| strava_1  | 2025-04-11 |  4.18 | 4:47       |      165 |     189 |        87 |              | high_hr              |                    |
| strava_1  | 2025-04-22 |  3    | 4:22       |      163 |     189 |        86 |              | high_hr              |                    |
| strava_1  | 2025-04-30 |  7.01 | 4:28       |      166 |     189 |        88 |              | high_hr              |                    |
| strava_10 | 2025-04-05 |  4.21 | 3:55       |      185 |     204 |        91 |              | high_hr              |                    |
| strava_11 | 2021-11-16 |  3.3  | 3:14       |      164 |     230 |        71 |              | speed_spike          |                    |
| strava_11 | 2023-11-11 |  3.01 | 3:28       |      201 |     230 |        87 |              | high_hr, speed_spike |                    |
| strava_12 | 2023-06-20 |  6.14 | 4:42       |      182 |     205 |        89 |              | high_hr              |                    |
| strava_12 | 2023-08-15 |  7.7  | 4:48       |      178 |     205 |        87 |              | high_hr              |                    |
| strava_12 | 2023-09-16 |  7    | 4:23       |      178 |     205 |        87 |              | high_hr              |                    |
| strava_12 | 2023-12-06 |  6.68 | 4:12       |      183 |     205 |        89 |              | high_hr              |                    |
| strava_12 | 2024-04-07 | 15.42 | 5:03       |      182 |     205 |        89 |              | high_hr              |                    |
| strava_12 | 2024-06-22 | 12.76 | 4:42       |      178 |     205 |        87 |              | high_hr              |                    |
| strava_12 | 2024-06-28 | 11.55 | 5:14       |      177 |     205 |        86 |              | high_hr              |                    |
| strava_12 | 2024-07-14 | 12    | 4:40       |      179 |     205 |        87 |              | high_hr              |                    |
| strava_12 | 2024-07-28 | 12.49 | 5:26       |      177 |     205 |        86 |              | high_hr              |                    |
| strava_12 | 2024-10-08 | 15.4  | 4:33       |      179 |     205 |        87 |              | high_hr              |                    |
| strava_12 | 2024-12-24 |  8.03 | 5:13       |      179 |     205 |        87 |              | high_hr              |                    |
| strava_12 | 2025-07-01 |  7.92 | 4:22       |      177 |     205 |        86 |              | high_hr              |                    |
| strava_12 | 2025-09-05 | 11.41 | 4:56       |      179 |     205 |        87 |              | high_hr              |                    |
| strava_13 | 2023-11-11 | 11.07 | 5:58       |      171 |     190 |        90 |              | high_hr              |                    |
| strava_13 | 2025-09-06 |  4    | 3:54       |      176 |     190 |        93 |              | high_hr              |                    |
| strava_14 | 2025-03-17 |  4.02 | 6:47       |      170 |     192 |        89 |              | high_hr              |                    |
| strava_14 | 2025-04-05 |  4.27 | 6:23       |      179 |     192 |        93 |              | high_hr              |                    |
| strava_14 | 2025-04-29 |  6.55 | 7:40       |      167 |     192 |        87 |              | high_hr              |                    |
| strava_14 | 2025-05-31 |  6.49 | 7:43       |      168 |     192 |        88 |              | high_hr              |                    |
| strava_14 | 2025-06-20 |  7.4  | 7:15       |      166 |     192 |        86 |              | high_hr              |                    |
| strava_14 | 2025-06-24 |  7.12 | 7:32       |      170 |     192 |        89 |              | high_hr              |                    |
| strava_14 | 2025-06-29 |  7.57 | 6:59       |      173 |     192 |        90 |              | high_hr              |                    |
| strava_14 | 2025-07-11 |  3.57 | 7:11       |      168 |     192 |        88 |              | high_hr              |                    |
| strava_14 | 2025-07-13 |  6.68 | 7:06       |      175 |     192 |        91 |              | high_hr              |                    |
| strava_14 | 2025-07-27 |  7.09 | 6:44       |      175 |     192 |        91 |              | high_hr              |                    |
| strava_14 | 2025-08-17 |  7.44 | 6:56       |      170 |     192 |        89 |              | high_hr              |                    |
| strava_14 | 2025-08-17 |  3.79 | 7:04       |      167 |     192 |        87 |              | high_hr              |                    |
| strava_14 | 2025-08-19 |  7.24 | 6:59       |      171 |     192 |        89 |              | high_hr              |                    |
| strava_14 | 2025-08-22 |  7.07 | 7:03       |      170 |     192 |        89 |              | high_hr              |                    |
| strava_14 | 2025-08-27 |  3.02 | 6:46       |      170 |     192 |        89 |              | high_hr              |                    |
| strava_14 | 2025-09-03 |  7.01 | 6:48       |      173 |     192 |        90 |              | high_hr              |                    |
| strava_14 | 2025-09-06 |  3.97 | 6:12       |      178 |     192 |        93 |              | high_hr              |                    |
| strava_15 | 2025-09-07 |  3.82 | 3:03       |      144 |     244 |        59 |              | speed_spike          |                    |
| strava_16 | 2025-08-14 | 17.39 | 6:29       |      167 |     190 |        88 |              | high_hr              |                    |
| strava_16 | 2025-08-18 |  6.5  | 5:31       |      171 |     190 |        90 |              | high_hr              |                    |
| strava_16 | 2025-08-23 | 12.02 | 5:53       |      167 |     190 |        88 |              | high_hr              |                    |
| strava_16 | 2025-08-29 | 14.44 | 6:04       |      167 |     190 |        88 |              | high_hr              |                    |
| strava_16 | 2025-09-03 |  5.81 | 5:34       |      165 |     190 |        87 |              | high_hr              |                    |
| strava_16 | 2025-09-06 | 17.61 | 5:44       |      174 |     190 |        92 |              | high_hr              |                    |
| strava_16 | 2025-09-27 | 15.3  | 5:50       |      172 |     190 |        91 |              | high_hr              |                    |
| strava_17 | 2025-09-06 |  3.88 | 3:47       |      177 |     210 |        84 |              | speed_spike          |                    |
| strava_6  | 2022-07-18 |  6.54 | 3:48       |      182 |     207 |        88 |              | high_hr              |                    |
| strava_7  | 2024-06-14 |  6.31 | 5:25       |      176 |     201 |        88 |              | high_hr              |                    |
| strava_7  | 2024-06-15 |  7.42 | 5:36       |      185 |     201 |        92 |              | high_hr              |                    |
