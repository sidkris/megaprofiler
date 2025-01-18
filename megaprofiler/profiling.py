import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score


def basic_profile_analysis(data):
    """Generate a basic profile of the dataset."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    profile = {
        "columns": data.columns.tolist(),
        "missing_values": data.isnull().sum().to_dict(),
        "data_types": data.dtypes.to_dict(),
        "unique_values": data.nunique().to_dict(),
        "summary_statistics": data.describe().to_dict(),
    }

    return profile




def categorical_data_analysis(data):
    """Perform analysis on categorical columns."""
    analysis = {}
    for col in data.select_dtypes(include=['object']).columns:
        analysis[col] = {
            'unique_values': data[col].unique().tolist(),
            'mode': data[col].mode()[0],
            'value_counts': data[col].value_counts().to_dict()
        }

    return analysis


    

def missing_data_heatmap(data):
    """Generate a heatmap of missing values in the dataset."""
    plt.figure(figsize = (10, 7))
    sns.heatmap(data.isnull(), cbar = False, cmap = "viridis")
    plt.title("-- MISSING DATA HEATMAP --")
    plt.show()




def memory_usage_analysis(data):
    """Profile the memory usage of each column in the dataset."""

    return data.memory_usage(deep = True)




def data_imbalance_analysis(data, target_column):
    """Detect class imbalance by examining the distribution of target labels."""

    return data[target_column].value_counts(normalize=True)



def data_skewness(data):
    """Compute skewness for numerical columns."""
    skew_summary = {}
    for col in data.select_dtypes(include=[np.number]):
        skew_summary[col] = stats.skew(data[col].dropna())

    return skew_summary



def data_kurtosis(data):
    """Compute kurtosis for numerical columns."""
    kurtosis_summary = {}
    for col in data.select_dtypes(include=[np.number]):
        kurtosis_summary[col] = stats.kurtosis(data[col].dropna())

    return kurtosis_summary




def validate(data, rules):

    """Validate the dataset against provided rules."""
    violations = []
    
    for rule in rules:
        column = rule.get('column')
        condition = rule.get('condition')
        message = rule.get('message', "Validation failed")
        
        if column not in data.columns:
            violations.append(f"Column '{column}' not found in data.")
            continue

        # Check for missing values
        if condition == 'no_missing':
            if data[column].isnull().sum() > 0:
                violations.append(message)

        # Check for data types
        elif condition == 'data_type':
            expected_type = rule.get('expected_type')
            if not pd.api.types.is_dtype_equal(data[column].dtype, expected_type):
                violations.append(message)

        # Check for value range
        elif condition == 'range':
            min_val, max_val = rule.get('min'), rule.get('max')
            if not data[column].between(min_val, max_val).all():
                violations.append(message)

    return violations
