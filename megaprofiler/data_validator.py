import pandas as pd

class DataValidator:

    def __init__(self, data):
        self.data = data

    @classmethod
    def validate(self, rules):
        """Validate the dataset against provided rules."""
        violations = []
        
        for rule in rules:
            column = rule.get('column')
            condition = rule.get('condition')
            message = rule.get('message', "Validation failed")
            
            if column not in self.data.columns:
                violations.append(f"Column '{column}' not found in data.")
                continue

            # Check for missing values
            if condition == 'no_missing':
                if self.data[column].isnull().sum() > 0:
                    violations.append(message)

            # Check for data types
            elif condition == 'data_type':
                expected_type = rule.get('expected_type')
                if not pd.api.types.is_dtype_equal(self.data[column].dtype, expected_type):
                    violations.append(message)

            # Check for value range
            elif condition == 'range':
                min_val, max_val = rule.get('min'), rule.get('max')
                if not self.data[column].between(min_val, max_val).all():
                    violations.append(message)

        return violations
