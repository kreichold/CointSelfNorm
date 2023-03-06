function results = Self_Normalized_Bootstrap_Inference(y,D,X,R1,r0,alpha,B,IC,qmin,qmax)
% Self-normalized test statistics proposed in Reichold and Jentsch (2022) 
% based on IM-OLS estimator of Vogelsang and Wagner (2014, Journal of Econometrics)
%--------------------------------------------------------------------------
% INPUTS:    y...                                   Tx1 left-hand side variable
%            D...                                   either empty array (d=0) or Txd matrix of deterministic regressors
%            X...                                   Txm matrix of stochastic regressors
%            R1...                                  sxm matrix of restrictions on coefficients corresponding to X
%            r0...                                  sx1 values of restrictions under the null
%            alpha...                               1x1 nominal size of tests
%            B...                                   1x1 number of bootstrap replications
%            IC...                                  either 'AIC' or 'BIC' to determine order of VAR
%            qmin...                                1x1 minimum order of VAR (>0)
%            qmax...                                1x1 maximum order of VAR
%-----------------------------------------------------------------------
% OUTPUTS:   theta_hat...                           (2m+d)x1 OLS estimate of theta in augmented partial sums regression
%            V_hat...                               (2m+d)x(2m+d) VCV estimator of theta_hat up to \Omega_{u\cdot v} and scaling
%            Z...                                   Tx(2m+d) regressor matrix in augmented and partial sums regression
%            eta_hat...                             1x1 self-normalizer based on IM-OLS residuals
%            eta_hat_ortho...                       1x1 self-normalizer based on orthogonalized IM-OLS residuals
%            eta_tilde_ortho...                     1x1 alternative self-normalizer based on orthogonalized IM-OLS residuals
%            tau_IM_eta_hat...                      1x1 test statistic based on eta_hat
%            tau_IM_eta_hat_ortho...                1x1 test statistic based on eta_hat_ortho
%            tau_IM_eta_tilde_ortho...              1x1 test statistic based on eta_tilde_ortho
%            boot_critval_tau_IM_eta_hat...         1x1 bootstrap critical value for tau_IM_eta_hat
%            boot_critval_tau_IM_eta_hat_ortho...   1x1 bootstrap critical value for tau_IM_eta_hat_ortho
%            boot_critval_tau_IM_eta_tilde_ortho... 1x1 bootstrap critical value for tau_IM_eta_tilde_ortho
%            qopt...                                1x1 optimal order of VAR determined by IC
%--------------------------------------------------------------------------
% Karsten Reichold, March 2023
%--------------------------------------------------------------------------
%% IM-OLS Estimation

    % Dimensions
    [T,m] = size(X);
    [~,d] = size(D);
    [s,~] = size(R1);
    
    % Partial sums, modified regression matrix and corresponding IM-OLS estimate:
    Sy = cumsum(y);
    SD = cumsum(D);
    SX = cumsum(X);
    Z = [SD,SX,X]; 
    theta_hat = Z\Sy;
    
%% Self-Normalized Test Statistics
    
    % Sample analog of asymptotic variance (up to a constant and without scaling) of estlarge:
    tempa = [zeros(1,d+2*m);Z];
    tempb = tempa(1:(end-1),:);
    C = ones(T,1)*sum(Z) - cumsum(tempb);
    V_hat = ((Z'*Z)\(C'*C))/(Z'*Z);
    
    % Self-normalizer based on IM-OLS residuals:
    Su_hat = Sy - Z*theta_hat;
    eta_hat = sum((T^(-1/2)*cumsum(diff(Su_hat))).^2)/T;
    
    % Self-normalizers based on orthogonalized IM-OLS residuals:
    Z_trafo = (1:T)'.*sum(Z,1) - cumsum(cumsum([zeros(1,d+2*m);Z(1:(T-1),:)]),1);
    Z_ortho = Z_trafo - Z*(Z\Z_trafo);
    Su_hat_ortho = Su_hat - Z_ortho*(Z_ortho\Su_hat);
    eta_hat_ortho = sum((T^(-1/2)*cumsum(diff(Su_hat_ortho))).^2)/T;
    eta_tilde_ortho = eta_hat_ortho + sum((cumsum(diff(Su_hat_ortho)) - sum(diff(Su_hat_ortho))).^2)/(T^2);
    
    % Self-normalized test statistics:
    R2 = [zeros(s,d),R1,zeros(s,m)];
    tau_IM_eta_hat = ((R2*theta_hat - r0)'/(R2*eta_hat*V_hat*R2'))*(R2*theta_hat - r0);
    tau_IM_eta_hat_ortho = ((R2*theta_hat - r0)'/(R2*eta_hat_ortho*V_hat*R2'))*(R2*theta_hat - r0);
    tau_IM_eta_tilde_ortho = ((R2*theta_hat - r0)'/(R2*eta_tilde_ortho*V_hat*R2'))*(R2*theta_hat - r0);
    
%% Bootstrap Critical Values

    % Residuals in original regression based on IM-OLS estimator:
    u_hat = y - [D,X]*theta_hat(1:(d+m),1);
    
    % First differences of X:
    v = [NaN(1,m);diff(X)];
    % Stacked error process (v_1 is unknown, start at t=2):
    w = [u_hat(2:end)';v(2:end,:)']; %(1+m)x(T-1)
   
    % Determine optimal q using AIC or BIC:
    qopt = IC_VAR(w,qmin,qmax,IC);

    % Yule-Walker coefficient estimates and residuals of VAR(qopt):
    [eps_hat,Phi_hat] = YuleWalker(w,qopt);
        % eps_hat: (m+1)x(T-1-qopt) centered VAR Sieve residuals
        % Phi_hat: (m+1)x((m+1)*qopt) matrix containing the qopt (m+1)x(m+1)-dim. VAR Sieve coefficient matrices

    % Create index matrix for the bootstrap:
        % note: In each bootstrap step draw T+k times from {1,...,T-1-qopt} with
        %       replacement, where k is the length of the burn-in period
    k = 100;
    % Set seed:
    %rng('default');
    rng(1);
    % Draw indices:
    idx_matrix = randi(T-1-qopt,B,T+k); % (Bx(T+k))-dim. matrix with integers from 1 to T-1-qopt

    bootdist_tau_IM_eta_hat = NaN(B,1);
    bootdist_tau_IM_eta_hat_ortho = NaN(B,1);
    bootdist_tau_IM_eta_tilde_ortho = NaN(B,1);
    
    parfor b = 1:B
        
        % Draw from the centered residuals:
        idx = idx_matrix(b,:);
        eps_star = eps_hat(:,idx); %#ok<PFBNS> %(m+1)x(T+k)
       
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
            theta_restr = theta_hat - (((Z'*Z)\R2')/((R2/(Z'*Z))*R2'))*(R2*theta_hat - r0);
            % Generate data under the null:
            y_star_H0 = [D,X_star]*theta_restr(1:(d+m),1) + u_star;
          
        % IM-OLS estimation as above, but now based on bootstrap data:
        Sy_star_H0 = cumsum(y_star_H0);
        SX_star = cumsum(X_star);
        Z_star = [SD,SX_star,X_star]; 
        theta_hat_star = Z_star\Sy_star_H0;
        
        % Construction of self-normalizers as above, but now based on bootstrap data:
            % Sample analog of asymptotic variance (up to a constant and without scaling) of estlarge:
            tempa_star = [zeros(1,d+2*m);Z_star];
            tempb_star = tempa_star(1:(end-1),:);
            C_star = ones(T,1)*sum(Z_star) - cumsum(tempb_star);
            V_hat_star = ((Z_star'*Z_star)\(C_star'*C_star))/(Z_star'*Z_star);
    
            % Self-normalizer based on IM-OLS residuals:
            Su_hat_star = Sy_star_H0 - Z_star*theta_hat_star;
            eta_hat_star = sum((T^(-1/2)*cumsum(diff(Su_hat_star))).^2)/T;
    
            % Self-normalizers based on orthogonalized IM-OLS residuals:
            Z_trafo_star = (1:T)'.*sum(Z_star,1) - cumsum(cumsum([zeros(1,d+2*m);Z_star(1:(T-1),:)]),1);
            Z_ortho_star = Z_trafo_star - Z_star*(Z_star\Z_trafo_star);
            Su_hat_ortho_star = Su_hat_star - Z_ortho_star*(Z_ortho_star\Su_hat_star);
            eta_hat_ortho_star = sum((T^(-1/2)*cumsum(diff(Su_hat_ortho_star))).^2)/T;
            eta_tilde_ortho_star = eta_hat_ortho_star + sum((cumsum(diff(Su_hat_ortho_star)) - sum(diff(Su_hat_ortho_star))).^2)/(T^2);
    
        % Construction of self-normalized test statistics as above, but now based on bootstrap data:
        bootdist_tau_IM_eta_hat(b,1) = ((R2*theta_hat_star - r0)'/(R2*eta_hat_star*V_hat_star*R2'))*(R2*theta_hat_star - r0);
        bootdist_tau_IM_eta_hat_ortho(b,1) = ((R2*theta_hat_star - r0)'/(R2*eta_hat_ortho_star*V_hat_star*R2'))*(R2*theta_hat_star - r0);
        bootdist_tau_IM_eta_tilde_ortho(b,1) = ((R2*theta_hat_star - r0)'/(R2*eta_tilde_ortho_star*V_hat_star*R2'))*(R2*theta_hat_star - r0);
     
    end
    
     % Return the floor((B+1)(1-alpha))-th largest elements of bootdist_tau_IM_eta_hat, bootdist_tau_IM_eta_hat_ortho, and bootdist_tau_IM_eta_tilde_ortho:
     boot_critval_tau_IM_eta_hat = boot_quantile(bootdist_tau_IM_eta_hat,alpha);
     boot_critval_tau_IM_eta_hat_ortho = boot_quantile(bootdist_tau_IM_eta_hat_ortho,alpha);
     boot_critval_tau_IM_eta_tilde_ortho = boot_quantile(bootdist_tau_IM_eta_tilde_ortho,alpha);
     
%% Return Results

results.theta_hat = theta_hat;
results.V_hat = V_hat;
results.Z = Z;
results.eta_hat = eta_hat;
results.eta_hat_ortho = eta_hat_ortho;
results.eta_tilde_ortho = eta_tilde_ortho;
results.tau_IM_eta_hat = tau_IM_eta_hat;
results.tau_IM_eta_hat_ortho = tau_IM_eta_hat_ortho;
results.tau_IM_eta_tilde_ortho = tau_IM_eta_tilde_ortho;
results.boot_critval_tau_IM_eta_hat = boot_critval_tau_IM_eta_hat;
results.boot_critval_tau_IM_eta_hat_ortho = boot_critval_tau_IM_eta_hat_ortho;
results.boot_critval_tau_IM_eta_tilde_ortho = boot_critval_tau_IM_eta_tilde_ortho;
results.qopt = qopt;
   
end