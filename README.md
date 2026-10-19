# Race Performance Predictor – Analysis and Prediction of Running Performance

## 🏃‍♂️ Project Description

The aim of this project is to develop advanced predictive models for forecasting race completion times based on training data. As part of the work, two different approaches were built and compared:

* **General model** – analysis of a large population of runners using basic training data
* **Personalized model** – detailed analysis of individual athletes’ training logs

---

## 🚀 Extension: installable package, evaluation and app

This fork turns the thesis into a package, an evaluation suite and a small web app (plan:
[docs/PLAN_ROZSZERZENIA.md](docs/PLAN_ROZSZERZENIA.md)).
**Based on:** *Race Performance Predictor* by Zuzanna Jarlaczyńska – the original notebooks, data and
thesis PDF are unchanged; logic ported from the notebooks is credited in the module docstrings.

### Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"          # + ".[deep]" for TensorFlow models, ".[kaggle]" for the download script

race-predictor predict "data/garmin_activities/Activities Jola.csv"   # CLI
race-predictor app                                                    # web app (Streamlit)
race-predictor evaluate                                               # baseline comparison
pytest && ruff check .
```

The app takes a Strava `activities.csv` or Garmin Connect CSV export and shows predicted 5 km / 10 km /
half / marathon times with an 80 % range, a comparison with the Riegel formula, a form-over-time curve and
the features driving the prediction. Ranges are empirical errors on athletes the model had not seen
(roughly ±15 %); the model is trained on only 170 races of 14 athletes, so treat predictions as indicative.

### What was built

| Stage | Where | Result |
|---|---|---|
| 1. Package | `sources/`, `processing/`, `domain/` | Strava/Garmin adapters (`@register_source`), cleaning pipeline, race detection, leak-free windows; tests + CI |
| 2. Baselines | `features/`, `models/`, `evaluation/` | Riegel, Ridge, boosting, thesis GRU; chronological and athlete-grouped evaluation → [results](docs/RESULTS_BASELINES.md) |
| 3. Transfer + tuning | `sources/kaggle.py`, `models/transfer.py`, `models/tuning.py` | Kaggle pretraining code (**real run pending the data download**), Optuna with nested CV → [tuning](docs/RESULTS_TUNING.md) |
| 4. Architectures + XAI | `models/sequence.py`, `evaluation/explain.py` | GRU vs TCN vs Transformer, SHAP and integrated gradients → [architectures](docs/RESULTS_ARCHITECTURES.md), [explainability](docs/RESULTS_EXPLAINABILITY.md) |
| 5. Embedded | `models/distill.py`, `models/tflite.py` | Distilled 4 KB int8 TFLite model, ~130× smaller than the GRU at equal accuracy → [results](docs/RESULTS_EMBEDDED.md) |
| 6. App | `predictor.py`, `cli.py`, `app/` | CLI + Streamlit app |
| Error analysis | `evaluation/breakdown.py`, `processing/pipeline.py` | Errors split into stand-alone races, triathlon runs and distances; median APE; filters for a ride logged as a run and HR-sensor spikes → [breakdown](docs/RESULTS_BREAKDOWN.md) |

### Headline findings (MAPE of race time on **unseen athletes**, 170 races)

| Model | MAPE |
|---|---|
| Ridge regression (used by the app) | 9.7 % |
| Riegel formula | 11.0 % |
| GRU / TCN / Transformer (3 seeds) | 15.0 % / 17.0 % / 17.7 % |
| Distilled MLP, int8 TFLite (3.7 KB) | 12.9 % |

With this little data the simple models win; the sequence models need the pretraining of stage 3 to pay off.
Most of the remaining error comes from triathlon runs (a quarter of the races). On **stand-alone races**, which are the
ones the app predicts, the Riegel formula is the most accurate on unseen athletes: 6.6 % MAPE (median 4.6 %) vs 9.0 %
for Ridge ([breakdown](docs/RESULTS_BREAKDOWN.md)).
Methodological notes (grouped CV by athlete, nested tuning, seed variance) are in the results files.

---

## 📁 Repository Structure

```text
Race_Performance_Predictor/
│
├── data/
│   ├── strava_activities/         # Exports from the Strava platform
│   └── garmin_activities/         # Exports from Garmin Connect
│
├── notebooks/
│   ├── marathon_model.ipynb       # Notebook for the general model
│   └── personalized_model.ipynb   # Notebook for the personalized model
│
├── Race_Performance_Predictor.pdf # Full thesis (PDF)
└── README.md                      # This file
```

---

## 🔬 Two Prediction Approaches

### 1. General Model (large-scale dataset)

* **Dataset:** 10.7 million activities from 72,842 athletes
* **Objective:** marathon time prediction based on basic training data
* **Architecture:** Bidirectional GRU
* **Performance:** NRMSE 9.85%, MAPE 15.52%

### 2. Personalized Model (richer data)

* **Dataset:** training logs of 20 athletes (Strava + Garmin)
* **Objective:** pace prediction for arbitrary distances (3–42 km)
* **Architecture:** hybrid GRU model with parallel processing of sequential and static features
* **Performance:** MAPE 7.34%, average error 0.3 min/km

---

## 🛠️ Technical Requirements

```bash
Python 3.8+
```

**Main libraries:**

* TensorFlow / Keras
* scikit-learn
* pandas, numpy
* matplotlib, seaborn

---

## 📊 Key Data Processing Steps

### General model:

* aggregation of user-level features (total distance, time, average pace),
* detection of marathon race results from training data,
* removal of anomalies and temporal data leakage,
* preparation of training sequences.

### Personalized model:

* unification of data from different platforms (Strava, Garmin),
* estimation of individual HRmax for each athlete,
* race identification using advanced heuristics,
* construction of training sequences using a 365-day window,
* extraction of static features (training aggregates, athlete attributes).

---

## 🧠 Model Architectures

### General Model (GRU):

* masking layer,
* Bidirectional GRU (64 units),
* LayerNormalization,
* additional GRU layers (32, 16 units),
* Dropout layers (0.3),
* Dense layers.

### Personalized Model (Hybrid GRU):

```text
Sequential input (232 steps × 11 features)
    ↓
Bidirectional GRU(16) → Dropout → BatchNorm
    ↓
Static input (12 features)
    ↓
Dense(32) → Dropout
    ↓
Concatenation → Dense(32) → Dropout → Dense(1)
```

---

## 📈 Results

| Model              | MAE       | RMSE      | MAPE   | Distance range |
| ------------------ | --------- | --------- | ------ | -------------- |
| General model      | 17.16 min | 23.07 min | 15.52% | Marathon only  |
| Personalized model | 4.6 min   | 11.8 min  | 7.34%  | 3–42 km        |

**Key findings:**

* the personalized model is 52.7% more accurate than the general model,
* total training distance is the most important feature (not time),
* sequential models (GRU) outperform traditional methods by ~40%,
* inclusion of heart rate data and cross-training significantly improves accuracy.

---

## 📋 Practical Applications

* race strategy planning (optimal starting pace),
* performance monitoring over time,
* training personalization,
* overtraining prevention through trend analysis.

---

## 🔮 Future Work

* expansion of the dataset with more athletes,
* integration of additional sensors (pulse oximeters, glucose monitors),
* modeling environmental conditions (weather, altitude, surface),
* web application for user data upload and analysis,
* injury risk prediction based on changes in running patterns.

---

## 📚 References

The most important sources are:
* El-Kassabi et al. (2020) – sequential models in sports performance prediction
* Ruiz-Mayo et al. (2016) – analysis of runners’ training data
* Riegel (1981) – classic race time prediction formula
* Dash (2024) – LSTM models for running performance forecasting
Those and many more can be found in the .pdf thesis file
---

## 👤 Author

This project was developed as part of a **Bachelor’s thesis in Data Science**.

**Contact:**

* LinkedIn: `https://www.linkedin.com/in/zjarlaczynska/`
* Email: `zuzia.jarl@gmail.com`

---


> **Note:** Due to data privacy constraints, example datasets included in the repository have been anonymized. Full, original datasets are available on Kaggle via links provided in the thesis.
