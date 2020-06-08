from scipy import stats


class AnalyticTesting:
    def __init__(self, test_conv, con_conv, ci=0.95):
        self.test_conv = test_conv
        self.con_conv = con_conv
        self.lift = self.test_conv - self.con_conv
        self.ci = ci
        self._z_score = stats.norm.isf((1 - self.ci) / 2)

    def update_sizes(
        self, total_size=None, con_ratio=None, test_size=None, con_size=None
    ):
        if (total_size != None) & (con_ratio != None):
            self.test_size = round(total_size * (1 - con_ratio))
            self.con_size = round(total_size * con_ratio)
        else:
            self.test_size = test_size
            self.con_size = con_size
        self._sd_test = (
            (self.test_conv * (1 - self.test_conv)) / self.test_size
        ) ** 0.5
        self._sd_con = ((self.con_conv * (1 - self.con_conv)) / self.con_size) ** 0.5

    def update_ci(self, new_ci):
        self.ci = new_ci
        self._z_score = stats.norm.isf((1 - self.ci) / 2)

    def _get_sd(self):
        """
        Calculate the standard deviation of the lift
        """
        self._sd = (
            (self.test_conv * (1 - self.test_conv)) / self.test_size
            + (self.con_conv * (1 - self.con_conv)) / self.con_size
        ) ** 0.5
        return self._sd

    def get_ci(self):
        """
        Calculate the confidence interval of the lift
        Assumptions: lift > 0, or test ratio > control ratio

        Returns:
        Lower bound and upper bound of the lift
        """
        lwr_bnd = self.lift - self._sd * self._z_score
        upr_bnd = self.lift + self._sd * self._z_score
        return lwr_bnd, upr_bnd

    def get_pvalue(self):
        """
        Calculate p-value of the lift between two ratios given their sample sizes
        """
        lift = -abs(self.test_conv - self.con_conv)
        p_value = 2 * stats.norm.cdf(lift, loc=0, scale=self._get_sd())
        return p_value

    def get_ci_test(self):
        """
        Calculate the confidence interval of the test conversion rate
        
        Returns:
        Lower bound and upper bound of the lift
        """
        lwr_bnd = self.test_conv - self._sd_test * self._z_score
        upr_bnd = self.test_conv + self._sd_test * self._z_score
        return lwr_bnd, upr_bnd

    def get_ci_con(self):
        """
        Calculate the confidence interval of the test conversion rate
        
        Returns:
        Lower bound and upper bound of the lift
        """
        lwr_bnd = self.con_conv - self._sd_con * self._z_score
        upr_bnd = self.con_conv + self._sd_con * self._z_score
        return lwr_bnd, upr_bnd


# if __name__ == "__main__":

