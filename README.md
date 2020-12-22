# CointSelfNorm
Self-Normalized Inference in Cointegrating Regressions

## Introduction
This repository contains MATLAB code to test general linear restrictions on `beta` in cointegrating regressions of the form `y = X*beta + u` using the self-normalized test statistic proposed in Reichold and Jentsch (2020), where `y` is a T-dimensional time series, `X` is a (T,m)-dimensional matrix of stochastic regressors and `u` is a T-dimensional vector of stationary error terms. Each column of `X` corresponds to one integrated regressor. The code allows to include a constant or a constant and polynomial time trends as (deterministic) regressors on the right-hand side (`y = D*gamma + X*beta + u`). <!-- However, the code does not allow to test restrictions on the coefficients corresponding to the deterministic regressors (yet). -->

The test decision can be based on either simulated asymptotically valid critical values or bootstrap critical values. In small to medium samples, the use of bootstrap critical values is recommended. To obtain bootstrap critical values, the procedure fits a finite order VAR to the resdiuals in the cointegrating regression and the first differences of the integrated regressors, as described in detail in Reichold and Jentsch (2020). The order of the VAR can be either specified in advance by the researcher or determined by information criteria (AIC or BIC) as described in Kilian and Lütkepohl (2017).

## Usage
Download the files and move them into your current working directory, `pwd`.

## Main Functions

### inference.m
This is the only function the researcher has to call manually. It tests general linear restrictions (specified by the researcher) on the coefficients corresponding to the stochastic regressors using the self-normalized test statistic. The test decision is based on either simulated asymptotically valid critical values or bootstrap critical values. The function always calls the following function:

+ **IMOLS.m**
This function estimates the coefficients corresponding to the stochastic regressors (and, if included, also to the deterministic regressors) using the IM-OLS estimator of Vogelsang and Wagner (2014). The function also computes the self-normalizer.

In addition, if the test decision is supposed to be based on bootstrap critical values, the following functions are required:

+ **IC_VAR.m**
If the order of the VAR is not specified in advance, this function determines the optimal order using either the AIC or the BIC, as described in Kilian and Lütkepohl (2017).

+ **VARsieve_YuleWalker.m**
This function uses the Yule-Walker estimator to fit a finite order VAR to the resdiuals in the cointegrating regression and the first differences of the integrated regressors.

+ **boot_quantile.m**
Given a number of bootstrap realizations of the self-normalized test statistic, this function returns the bootstrap critical value.

## Auxiliary Functions

### vec.m
This auxiliary function stacks the columns of a matrix.

## Illustration
To illustrate the procedure, we have generate data, `y`and `X` (available as MAT-files above), as described in Reichold and Jentsch (2020), with `beta=[1,1]'`, `rho1=rho2=0.6` and `T=100`. We now show how to use the `inference.m` function to test restrictions on `beta` using the self-normalized test. The illustration is also available as a MATLAB-Script in **illustration.m**.

<!--

````Matlab
% seed:
rng(9);

% sample size:
T = 100;

% level of error serial correlation and regressor endogeneity, respectively:
rho1 = 0.6;
rho2 = 0.6;

% coefficient vector corresponding to stochastic regressors:
beta = [1,1]';

% sample size plus burn-in period to ensure stationbarity of u and v:
k = T+101;

% preallocations:
e = randn(k,1);
nu1 = randn(k,1);
nu2 = randn(k,1);
u = zeros(k,1);
v1 = zeros(k,1);
v2 = zeros(k,1);

% data generating process:
for s = 2:k
  u(s) = rho1*u(s-1) + e(s) + rho2*(nu1(s)+nu2(s));
  v1(s) = nu1(s) + 0.5*nu1(s-1);
  v2(s) = nu2(s) + 0.5*nu2(s-1);
end

% delete burn-in period:
u = u((k-T+1):end);
v1 = v1((k-T+1):end);
v2 = v2((k-T+1):end);

% generate two integrated regressors and y:
x1 = cumsum(v1);
x2 = cumsum(v2);
X = [x1,x2];
y = X*beta + u;
````
-->

````Matlab
y = importdata('y.mat'); % (T,1)-dimensional
X = importdata('X.mat'); % (T,m)-dimensional
[T,m] = size(X);
````
To call

````Matlab
[est,testval,critval,testdec] = inference(y,X,deterreg,p,R,r,alpha,boot,B,q,IC,qmin,qmax)
````
the following inputs are required:

+ `deterreg` Specifies whether deterministic regressors should be included:
  + `'true'` if either a constant or a constant and polynomial time trends should be included
  + `'false'` if no deterministic regressors should be included
  
+ `p` Highest integer power of polynomial time trend:
  + `0` for a constant but without time trends
  + integer `k`, say, larger than zero for a constant and polynomial time trends up to order `k`
  + if `deterreg = 'false'` set `p = [];`
  
+ `R` Matrix of linearly independent restrictions on the coefficients corresponding to the stochastic regressors:
  + (s,m)-dimensional, s not larger than m
  
+ `r` Right-hand side of null hypothesis:
  + s-dimensional vector
  
The null hypothesis is thus given by `R*beta = r`.
  
+ `alpha` Nominal size of the test:
  + element in the interval (0,1)
  
+ `boot` Specifies whether bootstrap or simulated asymptotically valid critical values should be used:
  + `'true'` for bootstrap critical values
  + `'false'` for simulated asymptotically valid critical values
  
If simulated asymptotically valid critical values should be used, the following arguments can be replaced by `[]`.
  
+ `B` Number of bootstrap replications:
  + large integer such that `(B+1)(1-alpha)` is an integer
  
+ `q` Pre-specified order of the VAR:
  + integer larger or equal to one, but much smaller than sample size T
  + set `q = [];` if information criteria should be used to determine the order
  
If  `q` is pre-specified, the following arguments can be replaced by `[]`.
  
+ `IC` Choose the information criteria to determine the order of the VAR:
  + `'AIC'`
  + `'BIC'`
  
+ `qmin` Smallest possible order of the VAR:
  + integer larger or equal to one but much smaller than sample size T

+ `qmax` Largest possible order of the VAR:
  + integer larger than `qmin` but much smaller than sample size T

In this illustration we add an intercept and a linear time trend to the model. We use bootstrap critical values based on 1,999 replications to test the null hypothesis that both coefficients are equal to zero at 0.05 level. We choose the AIC to determine the optimal order of the VAR among all orders between one and the largest integer smaller than or equal to `T^(1/3)`.

````Matlab
deterreg = 'true';
p = 1;
R = eye(m,m);
r = zeros(m,1);
alpha = 0.05;
boot = 'true';
B = 1999;
q = [];
IC = 'AIC';
qmin = 1;
qmax = floor(T^(1/3));

[est,testval,critval,testdec] = inference(y,X,deterreg,p,R,r,alpha,boot,B,q,IC,qmin,qmax)

est =
   -0.3144
   -0.0382
    1.1798
    0.8723
    
testval =
  1.5597e+04
  
critval = 
  347.2060
  
testdec =
  1
````

We obtain the following outputs:

+ `est` The IM-OLS estimates of the intercept, the coefficient corresponding to the linear time trend and the coefficients corresponding to the stochastic regressors.

+ `testval` The realization of the self-normalized test statistic.

+ `critval` The (bootstrap) critical value.

+ `testdec` The test decision:
  + `0` if there is not enough evidence in the data to reject the null hypothesis
  + `1` if there is enough evidence in the data to reject the null hypothesis

We conclude that there is overwhelming evidence in the data that at least one of the two coefficients corresponding to the stochastic regressors is significantly different from zero at 0.05 level (remember, both coefficents are in fact equal to one).

## References

+ Kilian, L. Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis. Cambridge University Press, Cambridge.
+ Reichold, K., Jentsch, C. (2020). Accurate and (Almost) Tuning Parameter Free Inference in Cointegrating Regressions. *SFB 823 Discussion Paper*, TU Dortmund University.
+ Vogelsang, T.J., Wagner, M. (2014). Integrated Modified OLS Estimation and Fixed-b Inference for Cointegrating Regressions. *Journal of Econometrics* **178**, 741-760.
