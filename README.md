## experimental-design
An analysis toolkit focuses on design and analysis of A/B tests:
- what is the minimum size with enough statistical power?
- What is the confidence interval of a proportion or a mean?
- Is the difference between control and test conversion rates statistically and practically significant?

I introduced different approaches to answer those questions, including standard hypothesis tests, non-parametric tests and simulation.

I set up abstract classes to organize different types of metrics (proportion or mean) and tests (one sample vs two samples).


### Instructions

* Clone the repository

``` shell
git clone https://github.com/hanh-nguyen/experimental-design
cd experimental-design
```

* Install dependencies

``` shell
python -m pip install requirements.txt
```

* Run unit tests

``` shell
cd test
pytest
```