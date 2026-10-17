# Plan rozszerzenia projektu – propozycja do akceptacji

> **Status:** etapy 1–7 zaakceptowane (1–6 zrobione), rozszerzenie z sekcji 8 – propozycja do ustalenia z Zuzią.
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

**Stan na 17.10.2026:** etapy 1–6 są zrobione (Zuzia); transfer learning z etapu 3 czeka na uruchomienie na pełnym
zbiorze z Kaggle. Raport i prezentacje (etap 7) przechodzą na koniec harmonogramu z sekcji 8.

---

## 7. Pytania do Zuzi

1. Czy zgadzasz się na rozwijanie projektu w tym kierunku w ramach naszych zajęć? -tak
2. Jak chcesz być oznaczona – wystarczy sekcja „Based on” w README z linkiem do pracy? - tak
3. Czy możemy użyć danych Strava/Garmin z `data/` (zanonimizowane) w aplikacji i raportach? - tak
4. Czy masz linki do zbiorów na Kaggle (zbiór ogólny i spersonalizowany), żebyśmy mogli je pobrać skryptem? - tak, link: https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset, https://www.kaggle.com/datasets/zuskakluska/activities-garmin, https://www.kaggle.com/datasets/zuskakluska/strava-activities
5. Czy chcesz, żeby zmiany trafiały z powrotem do Twojego repo (pull requesty), czy rozwijamy je w forku? - rozwijamy na forku

---

## 8. Rozszerzenie: od predykcji do pomocy w przygotowaniu do startu

Mamy jeszcze ok. 4 miesiące (do połowy lutego 2027), więc proponujemy rozwinąć aplikację z „kalkulatora czasu”
w asystenta przygotowań do startu: **ile pobiegnę → czy mój cel jest realny → jak trenować do dnia startu**.

### 8.1. Co wynika z dotychczasowych wyników

Szczegóły: `docs/RESULTS_BREAKDOWN.md` (PR z analizą błędów).

- **Biegi w triathlonie to ¼ startów i główne źródło błędu.** Na zwykłych startach wzór Riegela myli się o 6,6 %
  (mediana 4,6 %), na biegach w triathlonie o 23 %. Typy startów trzeba rozdzielić.
- **Dla zwykłych startów nowych zawodników najlepszy jest Riegel**, poza maratonem (13 %). Modele uczone pomagają, gdy
  znają historię danego zawodnika i przy triathlonie.
- **Dane tętna wymagają czyszczenia.** Skoki czujnika zawyżały HRmax do 220–253 u 6 osób, a od HRmax zależy
  wykrywanie startów i każda analiza obciążenia.
- **Modele sekwencyjne przegrywają z prostymi przy 170 startach.** Szansą jest pretrening na Kaggle (etap 3).

### 8.2. Nowe etapy

Szczegółowa architektura (co powstaje, z czego korzysta, co zmienia, kolejność i zasady pracy):
[ARCHITEKTURA_ROZSZERZENIA.md](ARCHITEKTURA_ROZSZERZENIA.md).

| # | Etap | Zakres | Rezultat |
|---|---|---|---|
| 8 | Jakość danych | sprawdzenie startów dodanych po filtrze skoków tętna (+33), przegląd 50 „startów” o niestandardowym dystansie, poprawka HRmax jako domyślna | czystsze etykiety i HRmax, ponowne wyniki bazowe |
| 9 | Obciążenie treningowe | TRIMP z tętna i czasu; fitness / zmęczenie / forma (CTL / ATL / TSB); indywidualnie dopasowany model Banistera (impulse–response) | moduł `training_load/`, nowe cechy modeli (np. świeżość w dniu startu), wykres w aplikacji |
| 10 | Tryby startów | bieg uliczny / triathlon / korekta na przewyższenie (trail); model dobierany do trybu | lepsza dokładność dla każdego typu, wybór trybu w aplikacji |
| 11 | Niepewność | conformal prediction z podziałem po zawodnikach; sprawdzenie pokrycia przedziałów | „10 km: 44–47 min (80 %)” z gwarancją pokrycia zamiast przedziałów empirycznych |
| 12 | Planer celu | czy cel jest realny na daną datę; plan tygodni do startu oparty na zasadach (fazy, progresja ≤ 10 % / tydz., taper), tempa stref z przewidywanego czasu; plan tempa na dzień startu; eksport do kalendarza (`.ics`) | nowa zakładka aplikacji |
| 13 | Symulacje „co jeśli” | wpływ zmian treningu (kilometraż, długie wybiegania) na przewidywany czas – na modelu ML i modelu Banistera; jasne zastrzeżenie, że to korelacja, nie zalecenie | interaktywne scenariusze w aplikacji |
| 14 | Transfer learning – uruchomienie | pobranie zbioru Kaggle, pretrening, porównanie z Riegelem na zwykłych startach | odpowiedź, czy pretrening ratuje modele sekwencyjne |
| 15 | Raport i prezentacje | dawny etap 7 | slajdy, raport, demo |

Świadomie **nie** robimy uczenia ze wzmocnieniem do planowania treningów (brak danych, by je uczyć) ani analizy
tętna sekunda po sekundzie (eksporty mają tylko podsumowania treningów).

### 8.3. Jak to pokrywa przedmioty

| Etap | Advanced Python | Advanced ML |
|---|---|---|
| 8 | testy regresyjne potoku danych, walidacja | jakość etykiet, wpływ na ewaluację |
| 9 | moduł obliczeń na szeregach czasowych (pandas, generatory) | inżynieria cech, dopasowanie modelu fizjologicznego (optymalizacja parametrów) |
| 10 | wzorzec Strategy (model per tryb) | dobór modelu do podproblemu |
| 11 | – | conformal prediction, kalibracja niepewności |
| 12 | Builder / dataclasses / generatory tygodni planu, eksport `.ics`, testy reguł | – |
| 13 | interaktywny interfejs | wyjaśnienia kontrfaktyczne, analiza wrażliwości |
| 14 | strumieniowe przetwarzanie 10,7 mln wierszy | transfer learning |

### 8.4. Harmonogram (orientacyjny)

| Okres | Etapy |
|---|---|
| do połowy listopada | 8, 9; start 14 (pobranie danych i długie obliczenia w tle) |
| do połowy grudnia | 10, 11, wyniki 14 |
| do połowy stycznia | 12, 13 |
| do połowy lutego | 15 + zapas na poprawki |

### 8.5. Do ustalenia z Zuzią

1. **Podział etapów.** Martyna może wziąć wszystkie etapy 8–15. Czy chcesz któryś zrobić sama albo tylko robić review?
   Etapy 8 i 12–13 przebudowują `predictor.py` i aplikację (patrz architektura), więc daj znać, jeśli coś tam zmieniasz.
2. **Sposób pracy.** Osobny branch i pull request na etap (zamiast pushowania na `main`), żebyśmy nie robiły tego samego
   równolegle. Etap 5 zrobiłyśmy obie naraz.
3. **Domyślny model w aplikacji.** Riegel dla zwykłych startów (z korektą na maraton), czy zostajemy przy Ridge?
4. **Filtr skoków tętna.** Czy sprawdzasz 33 nowe starty (czy to prawdziwe zawody), zanim włączymy go domyślnie?
