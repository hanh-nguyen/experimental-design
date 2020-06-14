from scipy import stats
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


class AnalyticTesting:
    def __init__(self, alpha=0.05):
        """
        confidence: confidence level, = 1 - alpha (Type I error rate)
        """
        self.alpha = alpha

    def power(self, p_null, p_alt, n_null, n_alt):
        """
        Compute the power of detecting the difference in two populations with different proportion parameters, given a desired alpha rate.
        
        Input parameters:
            p_null    : base success rate under null hypothesis (i.e control response rate)
            p_alt     : desired success rate to be detected, must be larger than p_null (i.e test response rate)
            n_null    : number of observations in the control group
            n_alt     : number of observations in the test group
            
        Output value:
        power : Power to detect the desired difference, under the null.
        """
        se_null = np.sqrt(2 * p_null * (1 - p_null) / n_null)
        null_dist = stats.norm(loc=0, scale=se_null)
        crit_val = null_dist.ppf(1 - self.alpha)

        se_alt = np.sqrt(p_null * (1 - p_null) / n_null + p_alt * (1 - p_alt) / n_alt)
        alt_dist = stats.norm(loc=p_alt - p_null, scale=se_alt)
        beta = alt_dist.cdf(crit_val)

        return 1 - beta

    def experiment_size(self, p_null, p_alt, power=0.80):
        """
        Compute the minimum number of samples needed to achieve a desired power level for a given effect size.
        
        Input parameters:
            p_null     : base success rate under null hypothesis
            p_alt      : desired success rate to be detected
            power      : 1 - beta (beta: Type-II error rate)
        
        Output value:
            n : Number of samples required for each group to obtain desired power
        """
        # Get necessary z-scores and standard deviations (@ 1 obs per group)
        z_null = stats.norm.ppf(1 - self.alpha)
        z_alt = stats.norm.ppf(1 - power)
        sd_null = np.sqrt(2 * p_null * (1 - p_null))
        sd_alt = np.sqrt(p_null * (1 - p_null) + p_alt * (1 - p_alt))

        # Compute and return minimum sample size
        p_diff = p_alt - p_null
        n = ((z_null * sd_null - z_alt * sd_alt) / p_diff) ** 2
        return np.ceil(n)

    def confidence_interval(self, p, n):
        """
        Compute the confidence interval using an expected rate.
        If the observed rate is NOT within the interval, it is statistically different from the expected rate.
        """
        sd = np.sqrt(p * (1 - p) / n)
        me = sd * stats.norm.ppf(1 - self.alpha / 2)
        return (p - me, p + me)

    def experiment_size_statsmodels(self, p_null, p_alt, power=0.80):
        return NormalIndPower().solve_power(
            effect_size=proportion_effectsize(p_alt, p_null),
            alpha=self.alpha,
            power=power,
            alternative="larger",
        )

    def confidence_interval_diff(self, p_null, p_alt, n_null, n_alt):
        """
        Compute the confidence interval for the difference between two proportions p_null and p_alt
        If the interval does NOT include zero, the difference is statistically significant
        """
        p_pooled = ((p_null * n_null) + (p_alt * n_alt)) / (n_null + n_alt)
        sd_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_null + 1 / n_alt))
        me = sd_pooled * stats.norm.ppf(1 - self.alpha / 2)
        return (p_pooled - me, p_pooled + me)

    def pvalue(self, p_null, p_alt, n_null, n_alt):
        """
        Compute the p-value of the observed difference.
        If p-value < alpha, the difference is statistically significant.
        """
        p_pooled = ((p_null * n_null) + (p_alt * n_alt)) / (n_null + n_alt)
        sd_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_null + 1 / n_alt))
        z = (p_alt - p_null) / sd_pooled
        return (1 - stats.norm.cdf(z))

    def pvalue_sim(self, p_null, p_alt, n_null, n_alt, trials=100000):
        """
        Compute the p-value of the observed difference through simulation.
        If p-value < alpha, the difference is statistically significant.
        """
        p_pooled = ((p_null * n_null) + (p_alt * n_alt)) / (n_null + n_alt)
        result_null = np.random.binomial(n_null, p_pooled, trials)
        result_alt = np.random.binomial(n_alt, p_pooled, trials)
        samples = result_alt / n_alt - result_null / n_null
        return ((samples >= (p_alt - p_null)).mean())

def main():
    # print(AnalyticTesting().experiment_size(0.0017, 0.0024))
    # print(AnalyticTesting().confidence_interval(0.5, 345543 + 344660)) # (0.4988204138245942, 0.5011795861754058)
    # print(AnalyticTesting().confidence_interval(0.5, 345543 + 344660))


if __name__ == "__main__":
    main()
