%------------------------------------------------------------------------
% Karsten Reichold, December 2020
%------------------------------------------------------------------------

% Works with MATLAB Version R2019b or higher.

clear;
clc;

% load data
y = importdata('y.mat'); % (T,1)-dimensional
X = importdata('X.mat'); % (T,m)-dimensional
[T,m] = size(X);

% specify inputs
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

% call inference.m
tic;
[est,testval,critval,testdec] = inference(y,X,deterreg,p,R,r,alpha,boot,B,q,IC,qmin,qmax) %#ok<NOPTS>
toc;