function [est,testval,critval,testdec] = inference(y,X,deterreg,p,R,r,alpha,boot,B,q,IC,qmin,qmax)
%INFERENCE Summary of this function goes here
%   - allows for no deterministic regressors, constant only and constant and polynomial time trends
%   - only coefficients corresponding to stochastic regressors are restricted under the null hypothesis
%   - we do not need AT and invAT here and in IM-OLS function, since matrices scale out when computing test statistic
%
%   INPUTS:
%    y...          Tx1 dependent variable
%    X...          Txm regressor matrix
%    deterreg...   either 'true' or 'false', indicates whether deterministic regressors are present 
%    p...          1x1 either empty array of deterreg = 'false', or, if deterreg = 'true', 0 (intercept only) or 1,2,... for [1,t,t^2,...,t^p]
%    R...          sxm restriction matrix corresponding to stochastic regressors (s <= m)
%    r...          sx1 values of restrictions corresponding to stochastic regressors under the null
%    alpha...      1x1 nominal level of the test, (0 < alpha < 1)
%    boot...       either 'true' or 'false'
%    B...          1x1 number of bootstrap replicates or empty array
%    q...          predetermined order of the VAR or empty array
%    IC...         either 'AIC' or 'BIC' or empty array
%    qmin...       1x1 minimum order of VAR or empty array
%    qmax...       1x1 maximum order of VAR or empty array
%
%   OUTPUTS:
%    est...        mx1 estimate of cointegrating vector
%    testval...    1x1 realization of self-normalized test statistic
%    critval...    1x1 either asymptotic or bootstrap critical value
%    testdec...    1x1 test decision, either 0 (do not reject null) or 1 (reject null)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Karsten Reichold, December 2, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Deterministic regressors and scaling matrix:

    % Sample size and number of stochastic regressors:
    [T,m] = size(X);
    
    if strcmp(deterreg,'true')
    D = ((1:T)').^(0:p);
    else 
    D = [];
    end

    % Number of deterministic regressors
    [~,d] = size(D);

%% Integrated modified OLS estimation:

    % IM-OLS estimate in augmented regression and corresponding regressor matrix, 
    % sample analog of asymptotic variance (up to a constant) of estlarge 
    % and realization of self-normalizer:
    [estlarge,Z,V,eta] = IMOLS(y,D,X);
    % IM-OLS estimate of coefficients in original regression:
    est = estlarge(1:(d+m),1);
    
%% Realization of self-normalized test statistic:
    
    % Restrictions and scaling matrix:
    s = size(R,1);
    R2 = [zeros(s,d),R,zeros(s,m)];
    
    % Test statistic:
    testval = ((R2*estlarge-r)'/(R2*eta*V*R2'))*(R2*estlarge-r);

%% Generate critical value:

if strcmp(boot,'false') % generate asymptotically valid critical value
    
    % Number of replications:
    simnum = 10^4;
    % Number of observations:
    N = 10^4;
   
    Pi = eye(d+2*m,d+2*m); % without loss of generality
    % In the following we use the true R2, but we could also use, without
    % loss of generality, e.g., R2 = [eye(s,s),zeros(s,2*m-s)], as done to
    % create Table 1.
    
    asymdist = NaN(simnum,1);
    
    parfor reploop = 1:simnum
    
            % Setting Random Seed:    
            st = RandStream.create('mlfg6331_64','Seed',1000);
            RandStream.setGlobalStream(st);
            st.Substream = reploop;  

            dw = randn(1,N);
            wtemp = (N^(-1/2))*cumsum(dw,2); 
            dWv = randn(m,N);
            Wv = (N^(-1/2))*cumsum(dWv,2);
            
            if strcmp(deterreg,'true')
                deter = (((1:N)').^(0:p))';
                g = [diag(N^(-1)*(N.^(-(0:p))))*cumsum(deter,2);(1/N)*cumsum(Wv,2);Wv];
            else 
                g = [(1/N)*cumsum(Wv,2);Wv];
            end
            
            G = (1/N)*cumsum(g,2);
            G1 = G(:,end); 
            A1 = (1/N)*(g*g');
            Ztemp = A1\((1/N^(1/2))*sum((G1 - G).*dw,2)); 
            denominator = (1/N)*sum((wtemp - Ztemp'*g).^2);  
            A2 = (1/N)*((G1-G)*(G1-G)');
            Vtemp = ((Pi)')\((A1\A2)/A1)/Pi;
            numerator = (R2*(((Pi)')\Ztemp))'*((R2*Vtemp*R2')\(R2*(((Pi)')\Ztemp)));
            asymdist(reploop,1) = numerator/denominator;

    end
    
    critval = quantile(asymdist,1-alpha);   
    
else % generate bootstrap critical value
    
   % IM-OLS residuals and first difference of regressors in original regression
   u_hat = y - [D,X]*est;
    % Note that v_1 is unknown since x_0 is unknown. Start at t=2.
   v = [NaN(1,m);diff(X)]; 
   w = [u_hat(2:end)';v(2:end,:)']; %(m+1)x(T-1)
   
   % Determine optimal q using AIC or BIC:
   if isempty(q) == 1 % determine order of VAR using information criterion
      qopt = InformationCriteria(w,qmin,qmax,IC);
   else % use predetermined order q
      qopt = q;
   end
   
   % Yule-Walker coefficient estimates and residuals of VAR(qopt):
   [eps_hat,Phi_hat] = YuleWalker(w,qopt);
        %       eps_hat: (m+1)x(T-1-qopt) centered VAR Sieve residuals
        %       Phi_hat: (m+1)x((m+1)*qopt) matrix containing the qopt (m+1)x(m+1)-dim. VAR Sieve coefficient matrices
   
   % Create index matrix for the bootstrap:
        % note: In each bootstrap step draw T+k times from {1,...,T-1-qopt} with
        %       replacement, where k is the length of the burn-in period
    k = 100;
    
    % Set seed:
    rng('default');
    
    % Draw indices:
    idx_matrix = randi(T-1-qopt,B,T+k); % (Bx(T+k))-dim. matrix with integers from 1 to T-1-qopt
    
    bootdist = NaN(B,1);
    
    parfor b = 1:B
        
        % Draw from the centered residuals:
            idx = idx_matrix(b,:);
            eps_star = eps_hat(:,idx); %(m+1)x(T+k)
       
        % Generate bootstrap quantities w_t^*:
                % note: recall that we generate T+k values to take burn-in period into account
                %       and that we need qopt starting values (which are simply set to zero)
            w_star = NaN(m+1,T+k+qopt);
            w_star(:,1:qopt) = zeros(m+1,qopt); % set the qopt starting values to zero
            
            for t = 1:(T+k)
                Wlags = vec(w_star(:,t:(t+qopt-1))*flipud(eye(qopt))); % ((m+1)qopt)x1
                w_star(:,t+qopt) = Phi_hat*Wlags + eps_star(:,t); % (m+1)x(T+k+qopt)
            end
            
        % Partition w_star into u_star and v_star and delete starting values and burn-in period:
            u_star = w_star(1,(k+qopt+1):end)'; %Tx1
            v_star = w_star(2:end,(k+qopt+1):end)'; %Txm
            
        % Generate the bootstrap regressors:
            X_star = cumsum(v_star); 
        
        % Generate data UNDER THE NULL:
            % Compute the restricted IM-OLS estimator:
            estlarge_restr = estlarge - inv(Z'*Z)*R2'*inv(R2*inv(Z'*Z)*R2')*(R2*estlarge - r);
            est_restr = estlarge_restr(1:(d+m),1);
            % Generate data under the null:
            y_star_H0 = [D,X_star]*est_restr + u_star;
          
        % IM-OLS estimate in augmented bootstrap regression, bootstrap sample analog of asymptotic variance 
        % (up to a constant) of estlarge and bootstrap realization of self-normalizer:
        [estlarge_b,~,V_b,eta_b] = IMOLS(y_star_H0,D,X_star);
             
        % Bootstrap test statistics:
        bootdist(b,1) = ((R2*estlarge_b-r)'/(R2*eta_b*V_b*R2'))*(R2*estlarge_b-r);
     
    end
   
   % Return the floor((B+1)(1-alpha))-th largest element of bootdist:
   critval = boot_quantile(bootdist,alpha);
   
end

%% Test decision

testdec = double(testval > critval);
    

end

