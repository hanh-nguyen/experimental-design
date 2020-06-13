from scipy import stats
import numpy as np


class AnalyticTesting:
    def __init__(self, confidence=0.95):
        """
        confidence: confidence level, = 1 - alpha (Type I error rate)
        """
        self.confidence = confidence

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
        crit_val = null_dist.ppf(self.confidence)

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
        z_null = stats.norm.ppf(self.confidence)
        z_alt = stats.norm.ppf(1 - power)
        sd_null = np.sqrt(2 * p_null * (1 - p_null))
        sd_alt = np.sqrt(p_null * (1 - p_null) + p_alt * (1 - p_alt))

        # Compute and return minimum sample size
        p_diff = p_alt - p_null
        n = ((z_null * sd_null - z_alt * sd_alt) / p_diff) ** 2
        return np.ceil(n)


def main():
    print(AnalyticTesting().experiment_size(0.0017, 0.0024))


if __name__ == "__main__":
    main()
