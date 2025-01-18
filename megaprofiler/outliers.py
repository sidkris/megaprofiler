import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score


def zscore_outlier_analysis(data, threshold = 3):
    """Detect outliers using z-score."""
    data = data.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(data.select_dtypes(include = [np.number])))
    outliers = (z_scores > threshold).any(axis = 1)
    
    return data[outliers]


def iqr_outlier_analysis(data):
    """Detect outliers using IQR (Inter-Quartile Range)."""
    data = data.select_dtypes(include=[np.number])
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    return data[((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis = 1)]



def isolation_forest_analysis(data, contamination = 0.1):
    """Detect anomalies using Isolation Forest on numerical columns."""
    numerical_data = data.select_dtypes(include = [np.number]).dropna()
    isolation_forest = IsolationForest(contamination = contamination, random_state = 21)
    anomaly_labels = isolation_forest.fit_predict(numerical_data)
    return anomaly_labels
