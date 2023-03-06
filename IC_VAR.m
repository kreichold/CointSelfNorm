function [qopt] = IC_VAR(w,qmin,qmax,type)
% Calculates optimal lag length using either AIC or BIC
% INPUTS:    w...              kxT matrix, Observations for which VAR(q) is fitted
%            qmin...           integer, lower bound, >0
%            qmax...           integer, upper bound
%            type...           either 'AIC' or 'BIC'
%-----------------------------------------------------------------------
% OUTPUTS:   qopt...           integer, optimal lag length
%------------------------------------------------------------------------
% KR, May 2020
%------------------------------------------------------------------------

%%
% dimensions:
[k,T] = size(w);
% transform w into cell array to construct the matrix of lags easily:
w_cell = mat2cell(w,k,ones(T,1));

% crop w to evaluate AIC for each q on the same dataset: t=(qmax+1):T (see KL2017,p.56)
wnew = w(:,(qmax+1):T);
Tnew = size(wnew,2);

% preallocation:
ICvec = NaN(qmax-qmin+1,1);

for q = qmin:qmax
% create the regressor matrix, given by the matrix of the lags:
% note: matrix of indices is given by [qmax,qmax+1,...,T-1;qmax-1,qmax,...,T-2;...;qmax-(q-1),qmax-(q-2),...,T-q]
Xq = cell2mat(w_cell(toeplitz(qmax:-1:(qmax-(q-1)),qmax:(T-1))));
% OLS estimation:
Phi = wnew/Xq; %=(wnew*Xq')/(Xq*Xq');
% OLS residuals:
eps_hat = wnew - Phi*Xq;

% calculate AIC or BIC (note that they are based on Tnew - instead of T - OLS residuals):
    % compare KL(2017,p.55) but note that we don't have an intercept here
if isequal(type,'AIC')
    ICvec(q) = log(det((eps_hat*eps_hat')/Tnew)) + (2/Tnew)*(q*(k^2)); 
elseif isequal(type,'BIC')
    ICvec(q) = log(det((eps_hat*eps_hat')/Tnew)) + (log(Tnew)/Tnew)*(q*(k^2));
end

end

% calculate optimal lag length qopt:
[~,qopt_idx] = min(ICvec);
qopt = qmin + (qopt_idx - 1);
end