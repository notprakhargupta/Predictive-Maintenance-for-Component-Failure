# scripts/feature_engineering.py
import pandas as pd

# Load telemetry data
telemetry = pd.read_csv('data/PdM_telemetry.csv', parse_dates=['datetime'])

# Feature Engineering: Lag Features
def create_lag_features(df, lags, window):
    lag_cols = ['volt', 'rotate', 'pressure', 'vibration']
    for col in lag_cols:
        for lag in range(1, lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        df[f'{col}_rolling_min'] = df[col].rolling(window=window).min()
        df[f'{col}_rolling_max'] = df[col].rolling(window=window).max()
    return df

telemetry = create_lag_features(telemetry, lags=3, window=3)
telemetry.dropna(inplace=True)

# Save the processed dataset
telemetry.to_csv('data/processed_telemetry.csv', index=False)
