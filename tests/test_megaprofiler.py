import unittest
import pandas as pd
from megaprofiler.megaprofiler import MegaProfiler

class TestMegaProfiler(unittest.TestCase):

    def test_profiler(self):
        # Example DataFrame and rules
        df = pd.DataFrame({
            'age': [25, 30, 45, None],
            'name': ['Alice', 'Bob', 'Charlie', 'David']
        })

        rules = [
            {'column': 'age', 'condition': 'no_missing', 'message': 'Age column contains missing values.'},
            {'column': 'age', 'condition': 'range', 'min': 0, 'max': 120, 'message': 'Age must be a number between 0 and 120.'},
        ]

        # Generate profile and summary
        profile_report, rule_violations = MegaProfiler.summarize(df, rules)

        # Assert that the profiler works and returns the expected keys
        self.assertIn("columns", profile_report)
        self.assertIsInstance(rule_violations, list)

        print("Profile Report:")
        print(profile_report)
        print("Rule Violations:")
        print(rule_violations)


if __name__ == '__main__':
    TestMegaProfiler().test_profiler()
