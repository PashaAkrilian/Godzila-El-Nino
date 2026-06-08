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

# Team Members

### Team LogLoss

| Name | Role |
|--------|--------|
| **Dimas Pasha Akrilian** | Team Leader, Data Scientist, Model Development |
| **Shata Alwan Jalaluddin** | Data Engineering, Climate Feature Engineering |
| **Aaron Christian Daniel** | Model Validation, Explainable AI |

**Institution:** Universitas Negeri Semarang (UNNES)

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

# Model Development

Several forecasting approaches were evaluated:

| Model | R² Score |
|---------|---------:|
| ARIMA | -1.6806 |
| Prophet | -0.4919 |
| Random Forest | 0.7625 |
| LightGBM | 0.7842 |
| XGBoost | 0.8052 |

XGBoost achieved the strongest baseline performance and was selected for further optimization. :contentReference[oaicite:1]{index=1}

---

# Validation Strategy

To avoid temporal leakage, the project uses:

## Walk-Forward Validation

5-fold chronological validation:

```text
Fold 1 : 2001–2004 → 2004–2008
Fold 2 : 2001–2008 → 2008–2011
Fold 3 : 2001–2011 → 2011–2014
Fold 4 : 2001–2014 → 2014–2017
Fold 5 : 2001–2017 → 2017–2020
```

This methodology simulates real-world forecasting scenarios and improves model generalization. :contentReference[oaicite:2]{index=2}

---

# Model Performance

Final optimized model performance:

| Metric | Train | Test |
|----------|---------:|---------:|
| RMSE | 0.0104 | 0.0404 |
| MAE | 0.0080 | 0.0306 |
| R² Score | 0.9916 | 0.8257 |

The model successfully explains more than **82% of national drought variability** using climate-based predictors alone. :contentReference[oaicite:3]{index=3}

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
- Matplotlib
- Seaborn

---

# Key Contributions

### Dimas Pasha Akrilian
- Project Leadership
- Research Design
- XGBoost Development
- Bayesian Optimization

### Shata Alwan Jalaluddin
- Climate Data Engineering
- Feature Engineering
- Data Integration

### Aaron Christian Daniel
- Validation Framework
- Explainable AI
- Model Evaluation

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

---

# Authors

**Dimas Pasha Akrilian**  
**Shata Alwan Jalaluddin**  
**Aaron Christian Daniel**

Department of Statistics and Data Science  
Faculty of Mathematics and Natural Sciences  
Universitas Negeri Semarang (UNNES)
