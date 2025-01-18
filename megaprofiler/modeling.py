import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



def kmeans_clustering(data, n_clusters = 3):
    """Perform K-Means clustering on numerical columns."""
    numerical_data = data.select_dtypes(include=[np.number]).dropna()
    kmeans = KMeans(n_clusters = n_clusters, random_state = 21)
    clusters = kmeans.fit_predict(numerical_data)
    return clusters


def smote_balancing(data, target_column):
    """Balance data using SMOTE (for classification problems)."""
    numerical_data = data.select_dtypes(include = [np.number])
    X = numerical_data
    y = data[target_column]
    smote = SMOTE(random_state = 21)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def undersampling_balancing(data, target_column):
    """Balance data using Random Undersampling."""
    numerical_data = data.select_dtypes(include = [np.number])
    X = numerical_data
    y = data[target_column]
    undersampler = RandomUnderSampler(random_state = 21)
    X_res, y_res = undersampler.fit_resample(X, y)
    return X_res, y_res