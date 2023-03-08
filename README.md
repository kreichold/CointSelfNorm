# CointSelfNorm
Self-Normalized Bootstrap Inference in Cointegrating Regressions

## Introduction
This repository contains MATLAB code to test general linear restrictions on `beta` in cointegrating regressions of the form `y = D*delta + X*beta + u` using the self-normalized test statistics proposed in Reichold and Jentsch (2022). Here, `y` is a T-dimensional time series, `D` is a (T,d)-dimensional matrix of deterministic components, `X` is a (T,m)-dimensional matrix of integrated regressors, and `u` is a T-dimensional stationary error term.

To obtain VAR sieve bootstrap critical values, the procedure fits a finite order VAR to the regression resdiuals (obtained with the IM-OLS estimator) and the first differences of the integrated regressors. The order of the VAR is determined by information criteria (either AIC or BIC). For all details, please see Reichold and Jentsch (2022).

The VAR sieve bootstrap scheme may be of independent interest and can be extracted easily from the function **Self_Normalized_Bootstrap_Inference.m**.

## Usage
Download the files and move them into your current working directory, `pwd`.

## Main Function

+ **Self_Normalized_Bootstrap_Inference.m**
This is the only function that needs to be executed by the practitioner. It returns IM-OLS estimation results, realizations of self-normalized test statistics, and corresponding VAR sieve bootstrap critical values. 

## Auxiliary Functions

+ **IC_VAR.m**
This function determines the optimal order of the VAR using either AIC or BIC.

+ **YuleWalker.m**
This function uses the Yule-Walker estimator to fit a finite order VAR.

+ **boot_quantile.m**
Given a number of bootstrap realizations of a test statistic, this function returns the corresponding bootstrap critical value.

+ **vec.m**
This function stacks the columns of a matrix.

## Illustration
The script **example.m** provides a brief illustration on how to use the function **Self_Normalized_Bootstrap_Inference.m** in applications.

## Reference
Reichold, K., Jentsch, C. (2022). [A Bootstrap-Assisted Self-Normalization Approach to Inference in Cointegrating Regressions](https://doi.org/10.48550/arXiv.2204.01373). arXiv e-print 2204.01373.
