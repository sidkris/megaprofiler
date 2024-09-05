import pandas as pd

class MegaProfiler:

    def __init__(self, data):
    
        """Initialize with a pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.data = data

    @classmethod
    def profile(self):
    
        """Generate a basic profile of the dataset."""
        profile = {
            "columns": self.data.columns.tolist(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "data_types": self.data.dtypes.to_dict(),
            "unique_values": self.data.nunique().to_dict(),
            "summary_statistics": self.data.describe().to_dict(),
        }
        return profile


