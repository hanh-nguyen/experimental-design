from analyticdev import OneSampleProportion, TwoSamplesProportion
import pytest
import pandas as pd
import numpy as np
from scipy import stats
import os

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestTwoSamplesProportion:
    @classmethod
    def setup_class(cls):
        cls.twosamples = TwoSamplesProportion(0.1, 0.12)

    def test_power_1000(self):
        n = 1000
        self.twosamples.update_size(n, n)
        assert self.twosamples.power() == pytest.approx(0.4412, rel=1e-4)

    def test_power_3000(self):
        n = 3000
        self.twosamples.update_size(n, n)
        assert self.twosamples.power() == pytest.approx(0.8157, rel=1e-4)

    def test_power_5000(self):
        n = 5000
        self.twosamples.update_size(n, n)
        assert self.twosamples.power() == pytest.approx(0.9474, rel=1e-4)

    def test_minimum_size(self):
        assert self.twosamples.minimum_size() == 2863

    # def test_minimum_size_statsmodels(self):
    #     assert self.twosamples.minimum_size_statsmodels() == 2863


class TestStatisticalSignificanceData:
    @classmethod
    def setup_class(cls):
        cls.data = pd.read_csv("./data/statistical_significance_data.csv")
        cls.n_control = len(cls.data[cls.data["condition"] == 0])
        cls.p_control = cls.data.loc[cls.data["condition"] == 0, "click"].mean()
        cls.n_test = len(cls.data[cls.data["condition"] == 1])
        cls.p_test = cls.data.loc[cls.data["condition"] == 1, "click"].mean()
        cls.twosamples = TwoSamplesProportion(cls.p_control, cls.p_test)
        cls.twosamples.update_size(cls.n_control, cls.n_test)

    def test_pvalue(self):
        assert self.twosamples.pvalue() == pytest.approx(0.0394428219746)

    def test_pvalue_sim(self):
        assert self.twosamples.pvalue_sim() == pytest.approx(
            0.039785, rel=5e-3
        )  # relative tolerance = 0.005
