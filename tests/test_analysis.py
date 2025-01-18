import unittest
import pandas as pd
from megaprofiler.analysis import pearson_correlation_analysis, pca_analysis

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up the test data."""
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8],
            'C': [9, 10, 11, 12]
        })
    
    def test_pearson_correlation_analysis(self):
        """Test the Pearson correlation analysis function."""
        correlation_matrix = pearson_correlation_analysis(self.data)
        self.assertEqual(correlation_matrix.shape, (3, 3))  # Ensure the result is a 3x3 matrix

    def test_pca_analysis(self):
        """Test the PCA analysis function."""
        pca_result, explained_variance = pca_analysis(self.data, n_components=2)
        self.assertEqual(pca_result.shape, (4, 2))  # Should return a 4x2 matrix for 4 samples and 2 components
        self.assertEqual(len(explained_variance), 2)  # There should be two components for 2 PCA components

if __name__ == '__main__':
    unittest.main()
