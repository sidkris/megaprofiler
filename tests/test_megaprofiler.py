import sys
sys.path.append("megaprofiler")
import unittest
import pandas as pd
import numpy as np
from megaprofiler import MegaProfiler


class TestMegaProfiler(unittest.TestCase):

    def setUp(self):
        """Set up some sample datasets for testing."""
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 6, 7, np.nan, 9],
            'C': ['cat', 'dog', 'cat', 'cat', 'dog'],
            'D': [1.1, 2.2, np.nan, 4.4, 5.5]
        })

        self.prior_data = pd.DataFrame({
            'A': [2, 3, 4, 5, 6],
            'B': [5, 5, 7, 8, 10],
            'C': ['cat', 'dog', 'cat', 'cat', 'dog'],
            'D': [1.2, 2.1, 3.3, 4.5, 5.1]
        })

    def test_basic_profile_analysis(self):
        """Test the basic profile analysis."""
        profile = MegaProfiler.basic_profile_analysis(self.data)
        self.assertIn('columns', profile)
        self.assertIn('A', profile['columns'])
        self.assertEqual(profile['missing_values']['B'], 1)
    
    def test_pearson_correlation_analysis(self):
        """Test Pearson correlation analysis."""
        correlation_matrix = MegaProfiler.pearson_correlation_analysis(self.data)
        self.assertAlmostEqual(correlation_matrix.loc['A', 'B'], 1.0, places=2)
    
    def test_covariance_analysis(self):
        """Test covariance matrix generation."""
        covariance_matrix = MegaProfiler.covariance_analysis(self.data)
        self.assertIn('A', covariance_matrix.columns)
        self.assertIn('B', covariance_matrix.index)
    
    def test_zscore_outlier_analysis(self):
        """Test z-score based outlier detection."""
        outliers = MegaProfiler.zscore_outlier_analysis(self.data)
        self.assertTrue(outliers.empty)
    
    def test_iqr_outlier_analysis(self):
        """Test IQR-based outlier detection."""
        outliers = MegaProfiler.iqr_outlier_analysis(self.data)
        self.assertTrue(outliers.empty)
    
    def test_data_drift_analysis(self):
        """Test data drift analysis."""
        drift_summary = MegaProfiler.data_drift_analysis(self.data, self.prior_data)
        self.assertIn('A', drift_summary)
        self.assertAlmostEqual(drift_summary['A']['drift'], 1.0)
    
    def test_categorical_data_analysis(self):
        """Test analysis on categorical columns."""
        analysis = MegaProfiler.categorical_data_analysis(self.data)
        self.assertIn('C', analysis)
        self.assertEqual(analysis['C']['mode'], 'cat')
    
    def test_text_data_analysis(self):
        """Test basic text data analysis."""
        text_data = pd.DataFrame({'text': ['hello world', 'machine learning is cool']})
        tfidf_matrix = MegaProfiler.text_data_analysis(text_data, 'text')
        self.assertEqual(tfidf_matrix.shape, (2, 6))
    
    def test_data_imbalance_analysis(self):
        """Test data imbalance analysis."""
        imbalance = MegaProfiler.data_imbalance_analysis(self.data, 'C')
        self.assertAlmostEqual(imbalance['cat'], 0.6)
    
    def test_data_skewness(self):
        """Test skewness calculation."""
        skew_summary = MegaProfiler.data_skewness(self.data)
        self.assertAlmostEqual(skew_summary['A'], 0.0, places=1)
    
    def test_data_kurtosis(self):
        """Test kurtosis calculation."""
        kurtosis_summary = MegaProfiler.data_kurtosis(self.data)
        self.assertAlmostEqual(kurtosis_summary['A'], -1.3, places=1)
    
    def test_memory_usage_analysis(self):
        """Test memory usage profiling."""
        memory_usage = MegaProfiler.memory_usage_analysis(self.data)
        self.assertIn('A', memory_usage.index)
    
    def test_validate(self):
        """Test the dataset validation method."""
        rules = [
            {'column': 'A', 'condition': 'no_missing', 'message': 'A has missing values.'},
            {'column': 'B', 'condition': 'range', 'min': 4, 'max': 10, 'message': 'B out of range.'}
        ]
        violations = MegaProfiler.validate(self.data, rules)
        self.assertIn('B out of range.', violations)


if __name__ == '__main__':
    unittest.main()
