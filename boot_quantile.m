function [qboot] = boot_quantile(x,alpha)
%boot_quantile returns the floor((B+1)(1-alpha))-th largest element of the
%B-dimensional vector x. Note that (B+1)(1-alpha) should be an integer.
% INPUTS     x...           Bx1 vector of bootstrap realizations
%            alpha...       1xn vector of "nominal sizes"
%-----------------------------------------------------------------------
%OUTPUTS:    qboot...       1xn floor((B+1)(1-alpha))-th largest element of x
%------------------------------------------------------------------------
% KR, November 2020
%------------------------------------------------------------------------
B = size(x,1);
temp = sort(x,'ascend');
qboot = (temp(floor((B+1)*(1-alpha))))';
end

