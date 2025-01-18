import unittest
import pandas as pd
import numpy as np
from megaprofiler.analysis import pearson_correlation_analysis, pca_analysis

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up the test data."""
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8],
            'C': [9, 10, 11, 12]  # All numeric columns
        })
    
    def test_pearson_correlation_analysis(self):
        """Test the Pearson correlation analysis function."""
        correlation_matrix = pearson_correlation_analysis(self.data)
        # Ensure that the correlation matrix has the correct size based on numerical columns
        numerical_columns = self.data.select_dtypes(include=[np.number]).shape[1]
        self.assertEqual(correlation_matrix.shape, (numerical_columns, numerical_columns))

    def test_pca_analysis(self):
        """Test the PCA analysis function."""
        pca_result, explained_variance = pca_analysis(self.data, n_components=2)
        self.assertEqual(pca_result.shape, (4, 2))  # Should return a 4x2 matrix for 4 samples and 2 components
        self.assertEqual(len(explained_variance), 2)  # There should be two components for 2 PCA components

if __name__ == '__main__':
    unittest.main()
