import pandas as pd
import numpy as np
from data_validator import DataValidator as dv 
from report_generator import ReportGenerator as rg
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.feature_extraction.text import TfidfVectorizer

class MegaProfiler:

    @classmethod
    def profile(cls, data):

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        """Generate a basic profile of the dataset."""
        profile = {
            "columns": data.columns.tolist(),
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.to_dict(),
            "unique_values": data.nunique().to_dict(),
            "summary_statistics": data.describe().to_dict(),
        }
        return profile


    # @classmethod
    # def summarize(cls, data, rules):

    #     profile_ = cls.profile(data)
    #     profile_report = rg.generate_report(profile_)
    #     rule_violations = dv.validate(data, rules)

    #     return profile_report, rule_violations
    

    @classmethod
    def pearson_correlation_analysis(cls, data):
        """Returns the correlation matrix for numerical columns."""
        return data.corr(method = "pearson")
    

    @classmethod
    def covariance_analysis(cls, data):
        """Returns the covariance matrix for numerical columns."""
        return data.cov()
    

    @classmethod
    def missing_data_heatmap(cls, data):
        """Generate a heatmap of missing values in the dataset."""
        plt.figure(figsize = (10, 7))
        sns.heatmap(data.isnull(), cba = False, cmap = "viridis")
        plt.title("-- MISSING DATA HEATMAP --")
        plt.show()


    @classmethod
    def zscore_outlier_analysis(cls, data, threshold = 3):
        """Detect outliers using z-score."""
        z_scores = np.abs(stats.zscore(data.select_dtypes(include = [np.number])))
        outliers = (z_scores > threshold).any(axis = 1)
        
        return data[outliers]
    

    @classmethod
    def iqr_outlier_analysis(cls, data):
        """Detect outliers using IQR (Inter-Quartile Range)."""
        q1 = data.quantile(0.25)
        q3 = data.qunatile(0.75)
        iqr = q3 - q1

        return data[((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis = 1)]
    

    @classmethod
    def data_drift_analysis(cls, current_data, prior_data):
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
    

    @classmethod 
    def categorical_data_analysis(cls, data):
        """Perform analysis on categorical columns."""
        analysis = {}
        for col in data.select_dtypes(include=['object']).columns:
            analysis[col] = {
                'unique_values': data[col].unique().tolist(),
                'mode': data[col].mode()[0],
                'value_counts': data[col].value_counts().to_dict()
            }
        return analysis


    @classmethod
    def text_data_analysis(cls, data, text_column):
        """Perform basic NLP analysis on a text column."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data[text_column].fillna(''))
        return tfidf_matrix


    @classmethod
    def data_imbalance_analysis(cls, data, target_column):
        """Detect class imbalance by examining the distribution of target labels."""
        return data[target_column].value_counts(normalize=True)
    

    @classmethod
    def data_skewness(cls, data):
        """Compute skewness for numerical columns."""
        skew_summary = {}
        for col in data.select_dtypes(include=[np.number]):
            skew_summary[col] = stats.skew(data[col].dropna())
        return skew_summary
    

    @classmethod
    def data_kurtosis(cls, data):
        """Compute kurtosis for numerical columns."""
        kurtosis_summary = {}
        for col in data.select_dtypes(include=[np.number]):
            kurtosis_summary[col] = stats.kurtosis(data[col].dropna())
        return kurtosis_summary

