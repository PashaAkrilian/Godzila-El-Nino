# Forecasting Godzilla El Niño 2026 Impact on National Drought Risk Using XGBoost and Spatio-Temporal Climate Kinematics

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)
![Climate AI](https://img.shields.io/badge/Domain-Climate%20Intelligence-red.svg)

## Overview

This project develops an AI-powered Early Warning System (EWS) for forecasting the potential impact of a future **Godzilla El Niño 2026** event on drought conditions across Indonesia.

The proposed framework integrates:

- Climate Intelligence
- Spatio-Temporal Feature Engineering
- Explainable AI (XAI)
- Hyperparameter Tuning (random search)
- Quantile XGBoost Forecasting

The primary target is the **Standardized Drought Condition Index (SDCI)**, which is used to quantify national drought severity.

The key innovation of this research is the introduction of **Climate Kinematics Features**, transforming Oceanic Niño Index (ONI) observations into momentum and acceleration variables to detect early signs of extreme Pacific Ocean warming.

---

# Problem Statement

Extreme ENSO events (El Niño Southern Oscillation) can significantly reduce rainfall across Indonesia and trigger widespread drought conditions.

Traditional forecasting approaches often struggle because drought formation is:

- Non-linear
- Multi-variate
- Spatially dependent
- Influenced by long-term climate teleconnections

This project aims to build a robust machine learning framework capable of forecasting future drought risk before severe impacts occur.

---

# Dataset

The project integrates multi-source climate datasets from:

- CHIRPS Rainfall
- MODIS Vegetation Index (NDVI)
- MODIS Land Surface Temperature (LST)
- NOAA Oceanic Niño Index (ONI)
- Sea Surface Temperature (SST)

Data period:

```text
2000 – 2025
```

Data acquisition was performed through:

```text
Google Earth Engine (GEE)
```

---

# Climate Kinematics Innovation

One of the main contributions of this project is the development of:

## Climate Momentum

```text
v(t) = ONI(t) − ONI(t−1)
```

Measures how rapidly El Niño intensity changes.

## Climate Acceleration

```text
a(t) = v(t) − v(t−1)
```

Measures sudden warming escalation that may indicate the onset of a Godzilla El Niño event.

Additional temporal features include:

- Lag Features
- Rolling Statistics
- Exponential Moving Average (EMA)
- ONI Momentum
- ONI Acceleration

---

# Machine Learning Pipeline

```text
Multi-Sensor Climate Data
            ↓
Data Cleaning
            ↓
Time-Aware Interpolation
            ↓
Climate Feature Engineering
            ↓
Climate Kinematics Extraction
            ↓
Walk Forward Validation
            ↓
Hyperparameter Tuning (Random Search)
            ↓
Quantile XGBoost
            ↓
SHAP Explainability
            ↓
Drought Risk Forecasting
```

---

# Repository Structure

```text
R/godzila-el-nino.Rmd   Pipeline utama (R): data, EDA, fitur, komparasi model, XGBoost, evaluasi, proyeksi 2026
R/godzila-el-nino.html  Hasil render pipeline utama
R/hasil_R.json          Semua angka model (dipakai artikel dan slide)
R/analisis_R.Rmd        Analisis statistik lead-lag (OLS, Newey-West, Granger) + hasil render analisis_R.html
notebooks/              godzila-el-nino.ipynb: versi Python (Kaggle) dengan Optuna, pembanding hasil R
data/                   master_dataset_godzilla_elnino_2000_2025.csv
artikel_smds/           Artikel SMDS: main.tex/main.pdf (English), main_id.tex/main_id.pdf (Indonesia),
                        figs/ dan figs_en/, SMDSBib.bib, buat_figur.py (gambar dari data dan R/hasil_R.json)
docs/                   Laporan_Akhir.pdf, Slide_Presentasi.pdf (revisi dengan angka R)
docs/arsip/             Slide_Presentasi_asli_canva.pdf (ekspor Canva asli)
docs/slide_src/         revisi_canva.py (menyunting PDF asli memakai R/hasil_R.json), fonts/
```

# How to Run

- **R (utama):** buka `R/godzila-el-nino.Rmd` di RStudio lalu Knit (butuh paket xgboost, lightgbm, randomForest, forecast, zoo, ggplot2, jsonlite; `prophet` opsional). Seed tetap (42); dua render berurutan menghasilkan tabel perbandingan dan proyeksi identik.
- **Python (Kaggle, pembanding):** buka `notebooks/godzila-el-nino.ipynb` di Kaggle dengan dataset `godzila-el-nino-indonesia`, lalu Run All.
- **Artikel dan slide:** `cd artikel_smds && python3 buat_figur.py`, lalu `latexmk -pdf main.tex main_id.tex`. Slide: `cd docs/slide_src && python revisi_canva.py` (butuh pymupdf).

# Validation Strategy

- Split kronologis 80/20: train 2001-2020 (237 bulan), test 2021-2025 (60 bulan); test tidak disentuh saat tuning maupun early stopping.
- Tuning XGBoost: random search 50 trial (seed 42) pada search space yang sama dengan Optuna di notebook Python, dengan `TimeSeriesSplit(5)` pada data train. Optuna tidak tersedia di R.
- Early stopping (50 ronde) memakai 15% ekor data train sebagai validasi.
- **Lead 3 bulan:** fitur sampai bulan t-3 memprediksi SDCI bulan t.
- **Tanpa kebocoran target:** SDCI disusun dari NDVI, LST, dan curah hujan, sehingga variabel itu pada bulan yang sama tidak dipakai. Prediktor hanya 44 fitur turunan ONI/`sst_anomaly` (lag, rolling, EMA, momentum, akselerasi) dan musiman bulan.

# Model Performance (test 2021-2025, lead 3 bulan)

Hasil R (`R/godzila-el-nino.Rmd`), dibandingkan dengan notebook Python:

| Model | RMSE | MAE | R² test (R) | R² test (Python) |
|---|---:|---:|---:|---:|
| XGBoost (tuned) | 0.0935 | 0.0708 | **0.0698** | 0.0374 |
| LightGBM | 0.0971 | 0.0716 | -0.0036 | -0.0036 |
| Random Forest | 0.0981 | 0.0768 | -0.0253 | -0.0228 |
| Klimatologi (rata-rata bulanan) | 0.0993 | 0.0746 | -0.0505 | -0.0505 |
| XGBoost default | 0.1067 | 0.0789 | -0.2137 | -0.1697 |
| Prophet | 0.1153 | 0.0908 | -0.4171 | -0.4297 |
| ARIMA(1,1,1) | 0.1404 | 0.1189 | -1.0990 | -1.0989 |

Selisih XGBoost (tuned) antara R dan Python berasal dari prosedur tuning (random search vs Optuna TPE); selisih Random Forest, XGBoost default, dan Prophet berasal dari implementasi paket.

**Temuan utama:** setelah kebocoran target dibuang, ENSO (ONI/SST) tidak menambah daya prediksi SDCI nasional di atas pola musiman pada lead 3 bulan. R² model terbaik hanya 0.07, selisihnya terhadap klimatologi kecil, dan analisis lead-lag (`R/analisis_R.Rmd`) memberi kesimpulan serupa (OLS ENSO lag-3 + musiman R² -0.0554 vs klimatologi -0.0505).

Versi awal proyek melaporkan R² test 0.8257. Angka itu berasal dari fitur NDVI/LST/curah hujan bulan yang sama, yaitu komponen penyusun SDCI (regresi linier dari ketiganya saja menghasilkan R² = 1.0), sehingga bukan kemampuan prediksi.

**Proyeksi 2026 (skenario):** ONI naik linear 0.15 per bulan adalah asumsi, bukan forecast ONI. Puncak (SDCI terendah) jatuh pada September 2026 dengan median 0.399 (rentang 10%-90%: 0.36-0.54). Nilai itu di atas ambang kritis 0.3 tetapi di bawah rata-rata historis September (0.493); hanya 3 dari 26 bulan September (2002, 2015, 2019) yang lebih rendah. Karena model tidak jelas mengalahkan klimatologi, ini indikasi bersyarat skenario, bukan prakiraan tervalidasi (Python/Optuna memberi 0.529, rentang 0.41-0.68).

---

# Explainable AI

To improve interpretability, the project incorporates:

- SHAP (SHapley Additive exPlanations)
- Feature Contribution Analysis
- Spatial Risk Interpretation

This enables stakeholders to understand which climate variables most strongly influence drought predictions.

---

# Technologies Used

- R (xgboost, lightgbm, randomForest, forecast, prophet, lmtest, sandwich, ggplot2)
- Python (pandas, NumPy, XGBoost, Optuna, SHAP, scikit-learn, Matplotlib) untuk notebook pembanding
- Google Earth Engine

---

# Future Work

Potential improvements include:

- Transformer-based Climate Forecasting
- Graph Neural Networks
- Real-time Climate Monitoring Dashboard
- Integration with BMKG Forecast Systems

---

# License

This project is licensed under the MIT License.
