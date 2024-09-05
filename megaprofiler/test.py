import pytest
import pandas as pd
from megaprofiler import MegaProfiler as m

def test_profiler():
    data = pd.DataFrame({
        'age': [25, 30, None],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    profiler = m.profile(data)
    profile = profiler.profile()
    
    assert 'age' in profile['columns']
    assert profile['missing_values']['age'] == 1

test_profiler()