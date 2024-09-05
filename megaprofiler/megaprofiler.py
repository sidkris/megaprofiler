import pandas as pd
from data_validator import DataValidator as dv 
from report_generator import ReportGenerator as rg

class MegaProfiler:

    @classmethod
    def profile(self, data):

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        """Generate a basic profile of the dataset."""
        profile = {
            "columns": self.data.columns.tolist(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "data_types": self.data.dtypes.to_dict(),
            "unique_values": self.data.nunique().to_dict(),
            "summary_statistics": self.data.describe().to_dict(),
        }
        return profile


    @classmethod
    def summarize(self, data, rules):

        profile_ = self.profile(self.data)
        profile_report = rg.generate_report(profile_)
        rule_violations = dv.validate(data, rules)

        return profile_report, rule_violations