function [estlarge,Z,V,eta] = IMOLS(y,D,X)
% IM-OLS estimation (Vogelsang and Wagner, 2014, Journal of Econometrics)
% (ie, OLS estimation in augmented partial sum regression)
% and self-normalizer as in Reichold and Jentsch (2020).
%-----------------------------------------------------------------------
% INPUTS     y...               Tx1 dependent variable
%            D...               either empty array or Txd matrix of deterministic regressors
%            X...               Txm matrix of stochastic regressors
%-----------------------------------------------------------------------
% OUTPUTS:   estlarge...        (2m+d)x1 OLS estimated in augmented partial sum regression
%            Z...               Tx(2m+d) regressor matrix in augmented and partial sum regression
%            V...               (2m+d)x(2m+d) VCV estimator of estlarge up to \Omega_{u\cdot v} and scaling
%            eta...             1x1 self-normalizer
%------------------------------------------------------------------------
% Karsten Reichold, December 2020
%------------------------------------------------------------------------

    % Dimensions
    [T,m] = size(X);
    [~,d] = size(D);
    
    % Partial sums, modified regression matrix and corresponding IM-OLS estimate:
    Sy = cumsum(y);
    SD = cumsum(D);
    SX = cumsum(X);
    Z = [SD,SX,X]; 
    estlarge = Z\Sy;
    
    % Sample analog of asymptotic variance (up to a constant and without scaling) of estlarge:
    tempa = [zeros(1,d+2*m);Z];
    tempb = tempa(1:(end-1),:);
    C = ones(T,1)*sum(Z) - cumsum(tempb);
    V = ((Z'*Z)\(C'*C))/(Z'*Z);
    
    % Self-normalizer:
    Su_hat = Sy - Z*estlarge;
    eta = sum((cumsum(diff(Su_hat))).^2)/T^2;

end

