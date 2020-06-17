from abc import ABCMeta, abstractmethod
from scipy import stats
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


class ExperimentalDesign(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def pvalue(self):
        pass

    @abstractmethod
    def pvalue_sim(self):
        pass

    @abstractmethod
    def confidence_interval(self):
        pass


class OneSample(ExperimentalDesign):
    def __init__(self):
        ExperimentalDesign.__init__(self)


class TwoSamples(ExperimentalDesign):
    def __init__(self):
        ExperimentalDesign.__init__(self)

    @abstractmethod
    def power(self):
        pass

    @abstractmethod
    def minimum_size(self):
        pass


class OneSampleProportion(OneSample):
    def __init__(self, p_null, alpha=0.05):
        """
        p_null: expected proportion (under null hypothesis)
        """
        OneSample.__init__(self)
        self.p_null = p_null
        self.alpha = alpha

    def update_size(self, n_obs, n_success):
        """
        n_obs    : number of observations
        n_success: number of 'successes', i.e. number of heads when flipping coints, number of conversion in a targeting campaign
        """
        self.n = n_obs
        self.n_success = n_success
        self.sd = np.sqrt(self.p_null * (1 - self.p_null) * self.n)

    def pvalue(self):
        """
        To get a precise p-value, you should also perform a continuity correction, either adding or subtracting 0.5 to the total count before computing the area underneath the curve. 
        i.e. If we had 415 / 850 assigned to the control group, then the normal approximation would take the area to the left of  (415+0.5)/850=0.489  and to the right of  (435−0.5)/850=0.511
        """
        if self.n_success < self.p_null * self.n:
            n_alt = self.n_success + 0.5
        else:
            n_alt = self.n_success - 0.5
        z = (n_alt - self.p_null * self.n) / self.sd
        return 2 * stats.norm.cdf(z)

    def pvalue_sim(self, trials=100000):
        samples = np.random.binomial(self.n, self.p_null, trials)
        left = np.min(self.n_success, self.n * self.p_null)
        right = np.max(self.n_success, self.n * self.p_null)
        return np.logical_or(samples <= left, samples >= right).mean()

    def confidence_interval(self):
        """
        Compute the confidence interval using an expected rate.
        If the observed rate is NOT within the interval, it is statistically different from the expected rate.
        """
        sd = np.sqrt(self.p_null * (1 - self.p_null) / self.n)
        me = sd * stats.norm.ppf(1 - self.alpha / 2)
        return (self.p_null - me, self.p_null + me)


class TwoSamplesProportion(TwoSamples):
    def __init__(self, p_null, p_alt, alpha=0.05):
        """
        p_null  : base success rate under null hypothesis (i.e control response rate)
        p_alt   : desired success rate to be detected, must be larger than p_null (i.e test response rate)
        alpha   : type I error rate
        """
        TwoSamples.__init__(self)
        self.alpha = alpha
        self.p_null = p_null
        self.p_alt = p_alt
        self.p_diff = p_alt - p_null

    def update_size(self, n_null, n_alt):
        """
        n_null    : number of observations in the control group
        n_alt     : number of observations in the test group
        """
        self.n_null = n_null
        self.n_alt = n_alt
        self.p_pooled = ((self.p_null * self.n_null) + (self.p_alt * self.n_alt)) / (
            self.n_null + self.n_alt
        )
        self.sd_pooled = np.sqrt(
            self.p_pooled * (1 - self.p_pooled) * (1 / self.n_null + 1 / self.n_alt)
        )

    def pvalue(self):
        """
        Compute the p-value of the observed difference.
        If p-value < alpha, the difference is statistically significant.
        """
        z = (self.p_diff) / self.sd_pooled
        return 1 - stats.norm.cdf(z)

    def pvalue_sim(self, trials=100000):
        """
        Compute the p-value of the observed difference through simulation.
        If p-value < alpha, the difference is statistically significant.
        """
        result_null = np.random.binomial(self.n_null, self.p_pooled, trials)
        result_alt = np.random.binomial(self.n_alt, self.p_pooled, trials)
        samples = result_alt / self.n_alt - result_null / self.n_null
        return (samples >= (self.p_diff)).mean()

    def confidence_interval(self):
        """
        Compute the confidence interval for the difference between two proportions p_null and p_alt
        If the interval does NOT include zero, the difference is statistically significant
        """
        me = self.sd_pooled * stats.norm.ppf(1 - self.alpha / 2)
        return (self.p_pooled - me, self.p_pooled + me)

    def power(self):
        """
        compute the power to detect the desired difference between two success rates, under the null.
        power = 1 - beta (beta: Type-II error rate)
        """
        se_null = np.sqrt(2 * self.p_null * (1 - self.p_null) / self.n_null)
        null_dist = stats.norm(loc=0, scale=se_null)
        crit_val = null_dist.ppf(1 - self.alpha)
        se_alt = np.sqrt(
            self.p_null * (1 - self.p_null) / self.n_null
            + self.p_alt * (1 - self.p_alt) / self.n_alt
        )
        alt_dist = stats.norm(loc=self.p_diff, scale=se_alt)
        beta = alt_dist.cdf(crit_val)
        return 1 - beta

    def minimum_size(self, power=0.80):
        """
        Compute the minimum number of samples needed to achieve a desired power level for a given effect size.

        Output value:
            n : Number of samples required for each group to obtain desired power
        """
        # Get necessary z-scores and standard deviations (@ 1 obs per group)
        z_null = stats.norm.ppf(1 - self.alpha)
        z_alt = stats.norm.ppf(1 - power)
        sd_null = np.sqrt(2 * self.p_null * (1 - self.p_null))
        sd_alt = np.sqrt(
            self.p_null * (1 - self.p_null) + self.p_alt * (1 - self.p_alt)
        )

        # Compute and return minimum sample size
        n = ((z_null * sd_null - z_alt * sd_alt) / self.p_diff) ** 2
        return np.ceil(n)

    def minimum_size_statsmodels(self, power=0.80):
        return NormalIndPower().solve_power(
            effect_size=proportion_effectsize(self.p_alt, self.p_null),
            alpha=self.alpha,
            power=power,
            alternative="larger",
        )


def main():
    print(
        OneSampleProportion(0.5, 690203).confidence_interval()
    )  # (0.4988204138245942, 0.5011795861754058)
    print(OneSampleProportion(0.1, 106319).confidence_interval())


if __name__ == "__main__":
    main()
