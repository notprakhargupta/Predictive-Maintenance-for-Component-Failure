# scripts/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import anderson, probplot

# Load datasets
telemetry = pd.read_csv('data/PdM_telemetry.csv')
errors = pd.read_csv('data/PdM_errors.csv')
maint = pd.read_csv('data/PdM_maint.csv')
failures = pd.read_csv('data/PdM_failures.csv')
machines = pd.read_csv('data/PdM_Machines.csv')

# Display basic information about datasets
print(telemetry.info())
print(errors.info())
print(maint.info())
print(failures.info())
print(machines.info())

# EDA: Summary statistics
print(telemetry.describe())

# EDA: Distribution plots for telemetry data
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(telemetry['volt'], kde=True, ax=ax[0, 0])
sns.histplot(telemetry['rotate'], kde=True, ax=ax[0, 1])
sns.histplot(telemetry['pressure'], kde=True, ax=ax[1, 0])
sns.histplot(telemetry['vibration'], kde=True, ax=ax[1, 1])
plt.tight_layout()
plt.show()

# Normality tests
result = anderson(telemetry['volt'])
print(f'Anderson-Darling Test for volt: {result}')

probplot(telemetry['volt'], dist="norm", plot=plt)
plt.show()
