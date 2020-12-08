function [estlarge,Z,V,eta] = IMOLS(y,D,X)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

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

