## experimental-design
An analysis toolkit focuses on design and analysis of A/B tests:
- what is the minimum sample size with enough statistical power?
- What is the confidence interval of a proportion?
- Is the difference between two conversion rates or two average balances statistically and practically significant?

I introduced different approaches to answer those questions, including standard hypothesis tests, non-parametric tests and simulation.  
A unit test suite is also included.

__This project is a work in progress.__

### Instructions

* Clone the repository

``` shell
git clone https://github.com/hanh-nguyen/experimental-design
cd experimental-design
```

* Install dependencies

``` shell
python -m pip install -r requirements.txt
```

* Run unit tests

``` shell
cd test
pytest
```