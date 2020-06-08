from analyticdev import AnalyticTesting
import pytest
import pandas as pd
import numpy as np
from scipy import stats
import os

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestAnalyticTesting:
    @classmethod
    def setup_class(cls):
        cls.data = pd.read_csv("./data/statistical_significance_data.csv")
        cls.p_null = cls.data["click"].mean()
        cls.n_control = len(cls.data[cls.data["condition"] == 0])
        cls.p_control = cls.data.loc[cls.data["condition"] == 0, "click"].mean()
        cls.n_test = len(cls.data[cls.data["condition"] == 1])
        cls.p_test = cls.data.loc[cls.data["condition"] == 1, "click"].mean()

    def test_pvalue(self):
        se_p = np.sqrt(
            self.p_null * (1 - self.p_null) * (1 / self.n_control + 1 / self.n_test)
        )
        z = (self.p_test - self.p_control) / se_p
        pvalue = 1 - stats.norm.cdf(z)
        print(pvalue)
        assert pvalue == pytest.approx(0.0394428219746)
