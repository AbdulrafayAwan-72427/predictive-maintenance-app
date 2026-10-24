# Industrial Equipment Failure Predictor

An end-to-end Machine Learning web application designed to predict mechanical equipment failure using operational sensor telemetry. Built to address severe class imbalance via SMOTE and optimize recall using custom Precision-Recall threshold tuning.

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
