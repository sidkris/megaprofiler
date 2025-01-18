import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import statsmodels.api as sm


def pearson_correlation_analysis(data):
    """Returns the correlation matrix for numerical columns."""
    numerical_data = data.select_dtypes(include=[np.number])
    return numerical_data.corr(method = "pearson")
    

def covariance_analysis(data):
    """Returns the covariance matrix for numerical columns."""
    numerical_data = data.select_dtypes(include=[np.number])
    return numerical_data.cov()



def data_drift_analysis(current_data, prior_data):
    """Detect data drift by comparing basic statistics between two datasets."""
    drift_summary = {}
    for col in current_data.columns:
        if current_data[col].dtype != object:
            drift_summary[col] = {
                'current_mean': current_data[col].mean(),
                'previous_mean': prior_data[col].mean(),
                'drift': np.abs(current_data[col].mean() - prior_data[col].mean())
            }

    return drift_summary




def pca_analysis(data, n_components = 2):
    """Perform Principal Component Analysis on numerical columns."""
    numerical_data = data.select_dtypes(include = [np.number]).dropna()
    pca = PCA(n_components = n_components)
    pca_result = pca.fit_transform(numerical_data)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance



def multicollinearity_analysis(data):
    """Detect multicollinearity using Variance Inflation Factor (VIF) on numerical columns."""
    numerical_data = data.select_dtypes(include = [np.number]).dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = numerical_data.columns
    vif_data["VIF"] = [variance_inflation_factor(numerical_data.values, i) for i in range(numerical_data.shape[1])]
    return vif_data



def normality_test(data):
    """Perform a Shapiro-Wilk test for normality on numerical columns."""
    numerical_data = data.select_dtypes(include = [np.number])
    normality_results = {}
    for col in numerical_data.columns:
        normality_results[col] = stats.shapiro(numerical_data[col].dropna()).pvalue
    return normality_results




def ttest_analysis(data, numerical_feature_1, numerical_feature_2):
    """Perform a t-test between two numerical columns."""
    return stats.ttest_ind(data[numerical_feature_1].dropna(), data[numerical_feature_2].dropna())




def chi_square_test(data, categorical_column_1, categorical_column_2):
    """Perform a Chi-squared test between two categorical columns."""
    contingency_table = pd.crosstab(data[categorical_column_1], data[categorical_column_2])
    return stats.chi2_contingency(contingency_table)




def feature_importance_analysis(data, target_column):
    """Analyze feature importance using RandomForest."""
    numerical_data = data.select_dtypes(include = [np.number]).dropna()
    X = numerical_data
    y = data[target_column]
    rf = RandomForestClassifier(random_state = 21)
    rf.fit(X, y)
    feature_importances = pd.DataFrame(rf.feature_importances_, index = X.columns, columns = ['importance'])
    return feature_importances



def recursive_feature_elimination(data, target_column, n_features_to_select = 5):
    """Perform Recursive Feature Elimination (RFE) on numerical columns."""
    numerical_data = data.select_dtypes(include = [np.number])
    X = numerical_data
    y = data[target_column]
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    return X.columns[rfe.support_].tolist()  # Returns the selected features




def cross_validation_analysis(X, y, cv = 3):
    """Perform k-fold cross-validation and return the mean and std of scores."""
    return cross_val_score(RandomForestClassifier(random_state = 21), X, y, cv = cv).mean(), cross_val_score(RandomForestClassifier(random_state = 21), X, y, cv = cv).std()



def silhouette_analysis(data, cluster_labels):
    """Perform Silhouette Analysis for clustering."""
    numerical_data = data.select_dtypes(include = [np.number])
    score = silhouette_score(numerical_data, cluster_labels)
    return score  # Higher silhouette score indicates better-defined clusters



def tsne_analysis(data, n_components = 2, perplexity = 2):
    """Perform t-SNE dimensionality reduction on numerical columns."""
    numerical_data = data.select_dtypes(include = [np.number]).dropna()
    tsne = TSNE(n_components = n_components, perplexity = perplexity, random_state = 21)
    tsne_result = tsne.fit_transform(numerical_data)
    return tsne_result



def time_series_analysis(data, time_column, period = 1):
    """Perform basic time series analysis including decomposition and autocorrelation."""
    decomposition = sm.tsa.seasonal_decompose(data[time_column].dropna(), model='additive', period = period)

    return decomposition
    