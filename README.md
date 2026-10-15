# Race Performance Predictor – Analysis and Prediction of Running Performance

## 🏃‍♂️ Project Description

The aim of this project is to develop advanced predictive models for forecasting race completion times based on training data. As part of the work, two different approaches were built and compared:

* **General model** – analysis of a large population of runners using basic training data
* **Personalized model** – detailed analysis of individual athletes’ training logs

---

## 🚧 Extension (in progress)

This fork extends the thesis into an installable package and app (see [docs/PLAN_ROZSZERZENIA.md](docs/PLAN_ROZSZERZENIA.md)).
**Based on:** *Race Performance Predictor* by Zuzanna Jarlaczyńska – the original notebooks, data and thesis PDF are unchanged.

**Stage 1 (done): `src/race_predictor/`** – domain model, Strava/Garmin source adapters (`@register_source`),
a cleaning pipeline and race-detection strategies ported from `personalized_model.ipynb`, leak-free training windows, tests + CI.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check .
```

```python
import pandas as pd
from race_predictor.sources import load_directory
from race_predictor.processing import default_pipeline

df = pd.concat([load_directory("data/strava_activities"), load_directory("data/garmin_activities")])
df = default_pipeline().run(df)  # adds hr_max, is_triathlon, is_race
```

**Stage 2 (done): baselines + GRU + evaluation** – `features/` (one example per race), `models/` (Riegel, Ridge, LightGBM/HistGB, thesis GRU),
`evaluation/` (thesis chronological split and athlete-grouped CV). Results: [docs/RESULTS_BASELINES.md](docs/RESULTS_BASELINES.md).

```bash
python -m race_predictor.evaluation.runner [--with-gru]   # GRU needs `pip install -e ".[deep]"`, ~10 min
```

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
