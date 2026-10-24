# Industrial Equipment Failure Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictive-maintenance-app-e5l5xlrzb2kxr48lggepue.streamlit.app/)

An end-to-end Machine Learning web application designed to predict mechanical equipment failure using operational sensor telemetry. Built to address severe class imbalance via SMOTE and optimize recall using custom Precision-Recall threshold tuning.

## 🌐 Live Demo
You can test the live application here: [Predictive Maintenance App](https://predictive-maintenance-app-e5l5xlrzb2kxr48lggepue.streamlit.app/)

## Project Structure
predictive-maintenance-app/
├── data/
│   └── ai4i2020.csv
├── maintenance_model.pkl
├── scaler.pkl
├── optimal_threshold.pkl
├── train.py
├── app.py
└── requirements.txt

## How to Run Locally

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
