clear;
clc;

T = 100;
X = cumsum(randn(T,3));
y = X*ones(3,1) + randn(T,1);
%D = []; % no deterministic components
D = ones(T,1); % intercept only
%D = [ones(T,1),(1:T)']; % intercept and linear trend
R1 = eye(3);
r0 = ones(3,1);
alpha = 0.05;
B = 999;
IC = 'AIC'; % or 'BIC'
qmin = 1;
qmax = floor(T^(1/3));

results = Self_Normalized_Bootstrap_Inference(y,D,X,R1,r0,alpha,B,IC,qmin,qmax);