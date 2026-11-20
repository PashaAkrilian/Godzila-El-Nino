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
- Bayesian Optimization
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
Bayesian Optimization
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
notebooks/godzila-el-nino.ipynb   Notebook tunggal: pipeline Python + analisis R (sel terakhir)
R/analisis_R.Rmd                  Analisis R format R Markdown (Knit di RStudio)
data/                             master_dataset_godzilla_elnino_2000_2025.csv
artikel_smds/                     Artikel format SMDS: main.tex/main.pdf (English), main_id.tex/main_id.pdf (Indonesia),
                                  figs/ dan figs_en/ (gambar), SMDSBib.bib, buat_figur.py
docs/                             Laporan_Akhir.pdf, Slide_Presentasi.pdf (revisi), analisis_R.html
docs/arsip/                       Slide_Presentasi_asli_canva.pdf (ekspor Canva asli)
docs/slide_src/                   revisi_canva.py (menyunting PDF asli), hasil_kaggle_full.json, fonts/
```

# How to Run

- **Python (Kaggle):** buka `notebooks/godzila-el-nino.ipynb` di Kaggle dengan dataset `godzila-el-nino-indonesia`, lalu Run All. Seed tetap (42), jadi hasilnya identik di tiap run. Sel terakhir menjalankan analisis R lewat `Rscript`.
- **R (RStudio):** buka `R/analisis_R.Rmd` lalu Knit. Hasil render ada di `docs/analisis_R.html`.

# Validation Strategy

- Split kronologis 80/20: train 2001-2020, test 2021-2025 (test tidak disentuh saat tuning maupun early stopping).
- Tuning Optuna (50 trial, `TPESampler(seed=42)`) dengan `TimeSeriesSplit(5)` pada data train.
- Early stopping memakai 15% ekor data train sebagai validasi.
- **Lead 3 bulan:** fitur sampai bulan t-3 memprediksi SDCI bulan t.
- **Tanpa kebocoran target:** SDCI disusun dari NDVI, LST, dan curah hujan, sehingga variabel itu pada bulan yang sama tidak dipakai. Prediktor hanya turunan ONI/`sst_anomaly` (lag, rolling, EMA, momentum, akselerasi) dan musiman bulan.
- Hasil reproducible: 5 run berturut-turut di Kaggle menghasilkan angka identik.

# Model Performance (test 2021-2025, lead 3 bulan)

| Model | R² test |
|---|---:|
| XGBoost (Optuna) | 0.0374 |
| LightGBM | -0.0036 |
| Random Forest | -0.0228 |
| Klimatologi (rata-rata bulanan) | -0.0505 |
| XGBoost default | -0.1697 |
| Prophet | -0.4297 |
| ARIMA | -1.0989 |

**Temuan utama:** setelah kebocoran target dibuang, ENSO (ONI/SST) tidak menambah daya prediksi SDCI nasional di atas pola musiman pada lead 3 bulan. Analisis R di notebook yang sama memberi kesimpulan serupa (OLS ENSO lag-3 + musiman R² -0.0554 vs klimatologi -0.0505, identik dengan hasil Python).

Versi awal proyek melaporkan R² test 0.8257. Angka itu berasal dari fitur NDVI/LST/curah hujan bulan yang sama, yaitu komponen penyusun SDCI (regresi linier dari ketiganya saja menghasilkan R² = 1.0), sehingga bukan kemampuan prediksi.

**Proyeksi 2026 (skenario):** ONI naik linear 0.15 per bulan adalah asumsi, bukan forecast ONI. Puncak SDCI jatuh pada September 2026 dengan nilai 0.529 (rentang 10%-90%: 0.41-0.68). Nilai itu mendekati titik terendah musiman biasa dan berada di dalam rentang normal.

---

# Explainable AI

To improve interpretability, the project incorporates:

- SHAP (SHapley Additive exPlanations)
- Feature Contribution Analysis
- Spatial Risk Interpretation

This enables stakeholders to understand which climate variables most strongly influence drought predictions.

---

# Technologies Used

- Python
- Pandas
- NumPy
- XGBoost
- Optuna
- SHAP
- Scikit-Learn
- Google Earth Engine
- LightGBM, Prophet, Statsmodels
- R (lmtest, sandwich, ggplot2)
- Matplotlib
- Seaborn

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
