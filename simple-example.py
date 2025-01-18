import pandas as pd
import megaprofiler

# Sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': ['cat', 'dog', 'mouse', 'cat']
})

# Generate basic profile
profile = megaprofiler.basic_profile_analysis(data)
print(profile)

# Anomaly detection with z-score
outliers = megaprofiler.zscore_outlier_analysis(data)
print(outliers)

# Detect data drift (with prior data)
prior_data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [5, 6, 7],
})
drift_summary = megaprofiler.data_drift_analysis(data, prior_data)
print(drift_summary)
