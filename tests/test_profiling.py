import unittest
import pandas as pd
from megaprofiler.profiling import basic_profile_analysis, categorical_data_analysis

class TestProfiling(unittest.TestCase):

    def setUp(self):
        """Set up the test data."""
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8],
            'C': ['cat', 'dog', 'mouse', 'cat']
        })
    
    def test_basic_profile_analysis(self):
        """Test the basic profile analysis function."""
        profile = basic_profile_analysis(self.data)
        self.assertIn('columns', profile)
        self.assertIn('missing_values', profile)
        self.assertIn('data_types', profile)
        self.assertIn('unique_values', profile)
        self.assertIn('summary_statistics', profile)

    def test_categorical_data_analysis(self):
        """Test the categorical data analysis function."""
        analysis = categorical_data_analysis(self.data)
        self.assertIn('C', analysis)  # Check if 'C' is in the analysis
        self.assertIn('unique_values', analysis['C'])  # Ensure 'unique_values' is a key in the analysis for column 'C'
        self.assertIn('value_counts', analysis['C'])  # Check if 'value_counts' exists for 'C'

if __name__ == '__main__':
    unittest.main()
