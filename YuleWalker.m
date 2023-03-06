function [eps_hat,Phi_hat_YW] = YuleWalker(w,q)
% VAR Sieve estimated by Yule-Walker
%-----------------------------------------------------------------------
% INPUTS     w...            kxT matrix, w=[u_hat';v']
%            q...            1x1 order of the VAR
%-----------------------------------------------------------------------
% OUTPUTS:   eps_hat...      kx(T-q) centered VAR Sieve residuals
%            Phi_hat_YW...   kx(kq) matrix containing the q (kxk)-dim. VAR Sieve coefficient matrices
%------------------------------------------------------------------------
% Karsten Reichold, March 2023
%------------------------------------------------------------------------
%% Yule-Walker estimation in the regression of w_t on w_{t-1},...,w_{t-q}, t=q+1,...,T

% dimensions:
[k,T] = size(w);

% demean data:
    z = w - mean(w,2);
    
% add zero vector to create toeplitz matrix easily
    z_mod = [zeros(k,1),z];
    
% create cell array to allow for vector structure
    X = mat2cell(z_mod,k,ones(T+1,1));
    Z = cell2mat(X(toeplitz([1,zeros(1,q-1)],[1:T,zeros(1,q-1)])+1));
        % note: +1 ensures that we have the right indices: 1 corresponds to zero
        %       vector, 2 corresponds to x_1,..., T+1 corresponds to x_n
        
% create the toeplitz matrix of the (kxk) dimensional sample autocovariance matrices:
    G = (Z*Z')/T;
    
% stack the sample autocovariance matrices at lag 1,2,...,q:
    g = (cell2mat(X(toeplitz(zeros(1,q),0:(T-1))+1))*z')'/T;
    
% Yule-Walker estimates of the q (kxk)-marices:
    Phi_hat_YW = g/G; %kx(kq)
        % note: Phi_hat = [\hat{Phi}_1,...,\hat{Phi}_q], where \hat{Phi}_j is kxk

%% Computation Yule-Walker residuals:
% create the regressor matrix, given by the matrix of the lags:
        % note: corresponding matrix of indices is given by [q,q+1,...,T-1;q-1,q,...,T-2;...;1,2,...,T-q]
    w_cell = mat2cell(w,k,ones(T,1));
    LagMatrix = cell2mat(w_cell(toeplitz(q:-1:1,q:(T-1))));

% Yule-Walker residuals in the regression of w_t on w_{t-1},...,w_{t-q}, t=q+1,...,T:
    eps_hat_temp = w(:,(q+1):T) - Phi_hat_YW*LagMatrix; %kx(T-q)

% center residuals:
    eps_hat = eps_hat_temp - mean(eps_hat_temp,2); %kx(T-q)
end

