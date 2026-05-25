# Multi-Year Localized Air Pollution Analysis and Forecasting

> Point & Probabilistic Forecasting of daily PM₂.₅ concentrations using Statistical and Deep Learning Models.

**Course:** Applied Forecasting Methods (AFM) · DAU Gandhinagar  
**Team:** Harsh Jethwani · Anish Karmakar · Vrushil Panwala · Shah Trush

---

## Overview

A comprehensive end-to-end forecasting study using 21 years of daily PM₂.₅ data (2001–2021, 7,670 observations). The project spans the full model spectrum — from classical ARIMA-family models to recurrent deep learning and probabilistic quantile forecasting.

---

## Dataset

| Property | Value |
|---|---|
| Source | EPA AQS + Meteorological records |
| Period | 2001-01-01 → 2021-12-31 |
| Features | PM₂.₅, AQI (excluded), Relative Humidity, Temperature, Wind Speed |
| Split | 70% train / 10% val / 20% test |

> **Note:** AQI is excluded from all models — it is derived from PM₂.₅ and constitutes data leakage (r = 0.963).

---

## Models

**Statistical**
- AR, MA, ARMA, ARIMA, SARIMA, ARIMAX, SARIMAX

**Deep Learning**
- RNN, LSTM, GRU (point forecasting)
- RNN-Q, LSTM-Q, GRU-Q (quantile forecasting with pinball loss)

---

## Key Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naïve Lag-1 *(baseline)* | 2.52 | 3.635 | 0.113 |
| ARMA(2,1) *(best statistical)* | 2.20 | 3.203 | 0.311 |
| GRU *(base)* | 1.98 | 2.800 | 0.397 |
| **GRU\* *(tuned best point)*** | **1.62** | **2.170** | **0.531** |
| **GRU-Q\* *(best probabilistic)*** | **1.65** | **2.136** | **0.540** |

**Probabilistic calibration (GRU-Q\*):** PICP 80% = 0.8015 ✓ · PICP 90% = 0.9021 ✓ · MPIW 90% = 6.54 µg/m³

---

## Pre-processing Pipeline

1. **Outlier treatment** — Wind speed capped at 99th percentile (~10.5 m/s)
2. **Log-transform** — `log1p(PM₂.₅)` reduces skewness from 1.98 → 0.26
3. **Stationarity** — ADF test confirms d=1; working series = Δlog₁p(PM₂.₅)
4. **Anchored recovery** — `ŷ(t) = expm1(log_actual(t−1) + Δŷ(t))` prevents drift

---
## Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- PyTorch
- Scikit-learn
- Matplotlib

## Project Structure

```
├── FinalData.csv                          # Merged daily dataset
├── AFM_EDA_AirPollution.ipynb          # EDA · Pre-processing · Model Order Selection
├── AFM_Baseline_Statistical_Models.ipynb      # Baseline + Statistical models
├── PM25_Improved_OneDay_and_Recursive_Forecasting.ipynb   # RNN · LSTM · GRU + Quantile
├── PM25_Improved_Forecasting.ipynb   # Improved models · Residual MLP · GBT
└── Applied_Forecasting_Project.pdf        # Full project report
```

---

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn torch scipy
```

Run each notebook in order. Update `DATA_PATH` at the top of each notebook to point to `FinalData.csv`.

---

## Highlights

- 21 years of daily PM₂.₅ data
- Statistical + Deep Learning comparison
- Quantile forecasting with calibrated uncertainty intervals
- Best R²: 0.540 using GRU-Q*

## References

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- World Health Organization. (2021). *WHO Global Air Quality Guidelines: Particulate Matter (PM₂.₅ and PM₁₀), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide*. WHO.
- United States Environmental Protection Agency. (2023). *Air Quality System (AQS) Data Mart*. https://aqs.epa.gov/aqsweb/documents/data_mart_welcome.html