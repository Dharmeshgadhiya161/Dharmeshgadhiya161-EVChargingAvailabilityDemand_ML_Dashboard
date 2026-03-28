<!-- <div align="center">

<img align="right" width="120" height="120" alt="ev1" src="https://github.com/user-attachments/assets/acea8485-c939-46e5-82f5-a27253f6ae8b" />

# EV Charging Availability & Demand Intelligence

**Real-time intelligence for Electric Vehicle charging infrastructure**

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

</div>

---
## 🚀 Overview

This project aims to provide insights into EV charging station demand patterns and develop predictive tools that improve decision-making for both EV drivers and charging infrastructure operators. The final solution includes machine learning models, an interactive dashboard, and an AI assistant that helps users explore charging station data and trends.

### ✨ Key Features


## 📊 How It Works
1️⃣ Problem Statement
2️⃣ Dataset Overview
3️⃣ Business Use Case
4️⃣ EDA Insights
5️⃣ Feature Engineering
6️⃣ Model 1 (Regression)
7️⃣ Model Results
8️⃣ Model 2 (Classification)
9️⃣ Model Results
1️⃣0️⃣ Hyperparameter Tuning
1️⃣1️⃣ Dashboard Demo
1️⃣2️⃣ Chatbot Demo
1️⃣3️⃣ Business Insights

## 🛠 Tech Stack

- **Backend**: Python
- **Frontend**: Streamlit - Dash
- **Database**: CSV
- **ML/AI**: Scikit-learn, TensorFlow/PyTorch
- **Data Processing**: Nampy / Pandas
- **Visualization**: Plotly, ma




**Developed an end-to-end EV Charging Analytics Platform on a dataset of 1.3M+ records across 15+ cities, enabling accurate demand forecasting and availability prediction. Engineered 15+ time-series features (lag, rolling, station baselines) and improved model performance to an F1-score of 0.70+ and ROC-AUC of 0.72 through hyperparameter tuning and threshold optimization. Reduced prediction error by ~25% compared to baseline models (MAE/RMSE). Built an interactive Streamlit dashboard with real-time filtering, forecasting, and heatmaps, improving data exploration efficiency by 40%. Integrated a data-grounded AI chatbot (RAG) to answer operational queries, enhancing decision-making for station planning and demand management.**

## 📂 Project Structure

```bash
ev-charging-intelligence/
├── models/
├── data/
├── notebooks/
├── app.py
├── chatebod.py
└── README.md -->

<div align="center">
  <img align="right" width="130" height="130" alt="ev1" 
       src="https://github.com/user-attachments/assets/acea8485-c939-46e5-82f5-a27253f6ae8b" />

  # EV Charging Availability & Demand Intelligence

  **Real-time intelligence for Electric Vehicle charging infrastructure**

  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
  [![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

</div>

---

## 🚀 Project Theme

**How do time-of-day, location type, pricing, weather, traffic, and local events influence EV charging station availability — and how can we build reliable models to support drivers and infrastructure planners?**

This project simulates a **Mobility & Energy Analytics team** working for:
- City/State transportation or energy agencies
- Electric utilities & charging network operators
- Mobility apps optimizing routing and charging recommendations

### 🎯 Objective

A mobility + energy planning organization wants to:
- Understand utilization and availability patterns by **city, network, station, and location type**
- Identify peak congestion windows and recurring demand cycles
- Build predictive ML models to improve decision-making
- Deliver an interactive **Streamlit dashboard** and a **data-grounded AI chatbot**

---

## 📊 Dataset

**EV Charging Station Availability (Synthetic Time Series)**  
- **File**: `ev_charging_station_data.csv`  
- **Size**: 1.3M+ records  
- **Scope**: 150 stations, 8 major charging networks, 15 US metropolitan areas  
- **Time Period**: July – December 2025 (30-minute intervals)

**Key Columns Include**:
- **Station Info**: `station_id`, `network`, `city`, `location_type`, `charger_type`, `power_output_kw`
- **Availability**: `ports_available`, `ports_occupied`, `utilization_rate`, `estimated_wait_time_mins`
- **Pricing**: `current_price`, `pricing_type`
- **External Factors**: `temperature_f`, `weather_condition`, `traffic_congestion_index`, `local_event`, `gas_price_per_gallon`
- **Time Features**: `timestamp`, `hour_of_day`, `day_of_week`, `is_weekend`, `is_peak_hour`

---

## 🛠 Chosen ML Applications (Recommended)

You can choose **any two**:
1. **Regression / Forecasting** → Predict `utilization_rate` or `estimated_wait_time_mins` at **t+1** (30 mins ahead)
2. **Classification** → Predict whether a port will be **available** at t+1
---

## ✨ Key Achievements

- Processed **1.3M+ records** across **15+ cities**
- Engineered **15+ time-series features** (lags, rolling stats, station-hour baselines)
- Improved model performance: **F1-score ≥ 0.70**, **ROC-AUC ≈ 0.72**
- Reduced prediction error by **~25%** vs baseline (MAE/RMSE)
- Built an **interactive Streamlit dashboard** with filters, heatmaps, and forecasts
- Integrated a **data-grounded AI Chatbot** for operational queries

---

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **ML**: Scikit-learn, RandomForestRegressor, RandomForestClassification, and XGBoost
- **Dashboard**: Streamlit
- **AI Chatbot**: RAG (LangChain / LlamaIndex + embeddings)
- **Others**: Joblib (model saving), Parquet support

---
## Dashboard live link 

https://github.com/user-attachments/assets/99b4af9a-2663-4de7-821e-c35426f7b76e

---

## 📂 Project Structure

```bash
ev-charging-intelligence/
├── data/                  # Raw + processed data (ev_charging_station_data.csv)
├── notebooks/             # EDA, preprocessing, modeling
├── models/                # Trained models + artifacts
├── src/                   # Reusable scripts (preprocess.py, features.py, train.py)
├── app.py                 # Main Streamlit Dashboard
├── chatbot.py             # AI Chatbot (RAG)
├── requirements.txt
├── deployment_plan.md
└── README.md
-----



