# Predictive Maintenance for Component Failure

## Overview

This project predicts machine component failures using machine learning models such as Random Forest, XGBoost, and SVC. The goal is to predict whether a machine will fail in the near future (within 1 or 7 days) based on telemetry data, errors, and maintenance records.

## Data

The dataset is sourced from Kaggle and consists of five CSV files:

- `PdM_telemetry.csv`: Time series data of machine telemetry.
- `PdM_errors.csv`: Error logs for machines.
- `PdM_maint.csv`: Maintenance records.
- `PdM_failures.csv`: Machine failure records.
- `PdM_Machines.csv`: Metadata about the machines.

## Steps

1. **Data Collection and EDA**: Analyze data to understand distributions and correlations.
2. **Feature Engineering**: Create lag features and rolling statistics for predictive modeling.
3. **Model Training**: Train machine learning models to predict component failures.
4. **Model Evaluation**: Evaluate models using metrics such as recall and confusion matrix.
5. **Deployment**: Deploy a web app with Streamlit to provide real-time predictions.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/predictive_maintenance.git
cd predictive_
