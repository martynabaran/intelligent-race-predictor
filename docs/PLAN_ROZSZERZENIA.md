# Plan rozszerzenia projektu – propozycja do akceptacji

> **Status:** propozycja – czeka na zgodę autorki oryginalnego projektu (Zuzi).
> **Autorka propozycji:** Martyna Baran
> **Cel:** wspólny projekt zaliczeniowy na dwa przedmioty (AGH, AI & Data Analysis, sem. 2):
> **Advanced Python Programming** (dr K. Kluza) oraz **Advanced Machine Learning** (dr T. Kryjak).

---

## 1. Kontekst i podstawa

Projekt bazuje na pracy inżynierskiej *Race Performance Predictor* autorstwa Zuzi – notebookach
`marathon_model.ipynb`, `personalized_model.ipynb` oraz materiałach warsztatowych. Chcemy go rozwinąć
w pełnoprawną aplikację do przewidywania wyników biegowych na podstawie danych ze Stravy / Garmina.

**Zasady dotyczące autorstwa:**

- oryginalne notebooki, dane i PDF pracy pozostają w repozytorium bez zmian,
- README i prezentacje jasno wskazują, co pochodzi z pracy Zuzi, a co jest nowym wkładem,
- nowe moduły będą odwoływać się do pracy (cytowanie w README / docstringach tam, gdzie przenosimy logikę).

---

## 2. Główna idea

**Model ogólny wytrenowany na dużym zbiorze z Kaggle (10,7 mln aktywności) → dostrojony (transfer learning)
do danych konkretnego zawodnika ze Stravy / Garmina → opakowany w aplikację, do której użytkownik
wgrywa swój eksport i dostaje przewidywane czasy na 5 km / 10 km / półmaraton / maraton.**

To łączy oba istniejące podejścia z pracy (model ogólny + spersonalizowany) w jeden spójny system.

---

## 3. Część: Advanced Python Programming

Wymagania z sylabusa: kod „Pythonic”, generatory, context managery, dekoratory, testy jednostkowe,
obsługa wyjątków, OOP, refaktoryzacja, wzorce projektowe, Python do analizy danych, projekt grupowy + prezentacja.

| Element | Opis | Pokrywa |
|---|---|---|
| Pakiet `src/race_predictor/` | `pyproject.toml`, type hints, ruff, struktura modułów zamiast notebooków | refaktoryzacja, Pythonic code |
| Model domenowy | `Activity`, `Athlete`, `Race` jako dataclasses | OOP, struktury danych |
| Adaptery źródeł danych | abstrakcja `ActivitySource` + `StravaCsvSource`, `GarminCsvSource` (opcjonalnie `.fit`); rejestracja przez dekorator `@register_source` | wzorce Adapter / Strategy / Registry, dekoratory |
| Pipeline czyszczenia | łańcuch kroków (`RemoveAnomalies`, `ValidateHeartRate`, `EstimateHRMax`, …) | Pipeline / Chain of Responsibility |
| Wykrywanie startów | wymienne strategie (heurystyka HR, dystanse startowe) | Strategy |
| Przetwarzanie dużych danych | strumieniowe czytanie zbioru Kaggle w kawałkach, generator okien sekwencji | generatory |
| Infrastruktura | `timer()`, ładowanie modelu, śledzenie eksperymentów | context managery |
| Błędy | własne wyjątki, np. `UnsupportedExportFormatError` | obsługa wyjątków |
| Testy | `pytest`: parsery, heurystyki, brak wycieku danych train/test | testy jednostkowe |
| CI | GitHub Actions: lint + testy | inżynieria oprogramowania |
| Interfejs | CLI (`typer`) + aplikacja **Streamlit** | Python do analizy danych |

**Aplikacja Streamlit:** upload eksportu → przewidywane czasy na standardowe dystanse (z przedziałem
niepewności) → porównanie z formułą Riegela → wykres formy w czasie → wyjaśnienie, co wpłynęło na predykcję.

---

## 4. Część: Advanced Machine Learning

Wymagania z sylabusa: zaawansowane metody ML, optymalizacja parametrów, ekstrakcja cech, złożone pipeline'y,
krytyczna ocena wyników; tematy wykładu m.in. transformery, transfer learning, knowledge distillation,
explainability, modele lekkie / embedded.

1. **Baseline'y** – formuła Riegela, regresja liniowa, LightGBM.
2. **Modele sekwencyjne** – obecny GRU (z pracy) vs TCN vs mały Transformer encoder.
3. **Transfer learning** – pretrening na zbiorze Kaggle, fine-tuning na danych Strava/Garmin;
   porównanie z modelem uczonym od zera.
4. **Optymalizacja hiperparametrów** – Optuna.
5. **Rzetelna ewaluacja** – walidacja krzyżowa z podziałem po zawodnikach (`GroupKFold`), analiza błędów
   wg dystansu i typu zawodnika.
6. **Explainability** – SHAP / integrated gradients: które tygodnie treningu i cechy wpływają na wynik.
7. **Model „embedded”** – knowledge distillation do małego modelu + kwantyzacja TFLite int8;
   porównanie rozmiaru, latencji i dokładności (scenariusz: predykcja na zegarku sportowym).

**Framework:** TensorFlow / Keras (ciągłość z pracą Zuzi, łatwy eksport do TFLite).

---

## 5. Docelowa struktura repozytorium

```text
intelligent-race-predictor/
├── data/                      # bez zmian (+ skrypt pobierania zbioru Kaggle)
├── notebooks/                 # oryginalne notebooki – bez zmian
├── workshops/                 # bez zmian
├── docs/                      # plan, raporty z eksperymentów
├── src/race_predictor/
│   ├── domain/                # Activity, Athlete, Race
│   ├── sources/               # adaptery Strava / Garmin / Kaggle
│   ├── processing/            # pipeline czyszczenia, wykrywanie startów, cechy
│   ├── models/                # baseline'y, GRU/TCN/Transformer, transfer, destylacja
│   ├── evaluation/            # metryki, CV, explainability
│   ├── cli.py
│   └── app/                   # Streamlit
├── experiments/               # notebooki z wynikami eksperymentów ML
├── tests/
├── pyproject.toml
└── README.md
```

---

## 6. Etapy

| # | Etap | Rezultat |
|---|---|---|
| 1 | Refaktoryzacja notebooków do pakietu, adaptery i pipeline | dane Strava/Garmin wczytywane z `src/`, testy |
| 2 | Baseline'y + przeniesienie GRU, ewaluacja po zawodnikach | tabela wyników bazowych |
| 3 | Pretrening na Kaggle + transfer learning, Optuna | porównanie: od zera vs fine-tuning |
| 4 | TCN / Transformer, SHAP | porównanie architektur, wykresy wyjaśnień |
| 5 | Destylacja + TFLite | tabela rozmiar / latencja / dokładność |
| 6 | Aplikacja Streamlit + CLI, CI, README | działające demo |
| 7 | Raport i prezentacje na oba przedmioty | slajdy + dokumentacja |

---

## 7. Pytania do Zuzi

1. Czy zgadzasz się na rozwijanie projektu w tym kierunku w ramach naszych zajęć? -tak
2. Jak chcesz być oznaczona – wystarczy sekcja „Based on” w README z linkiem do pracy? - tak
3. Czy możemy użyć danych Strava/Garmin z `data/` (zanonimizowane) w aplikacji i raportach? - tak
4. Czy masz linki do zbiorów na Kaggle (zbiór ogólny i spersonalizowany), żebyśmy mogli je pobrać skryptem? - tak, link: https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset, https://www.kaggle.com/datasets/zuskakluska/activities-garmin, https://www.kaggle.com/datasets/zuskakluska/strava-activities
5. Czy chcesz, żeby zmiany trafiały z powrotem do Twojego repo (pull requesty), czy rozwijamy je w forku? - rozwijamy na forku
