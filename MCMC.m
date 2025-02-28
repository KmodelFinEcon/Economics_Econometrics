%Monte-carlo-Markov-chain OLS function by KT

%distribution used in complex derivative pricing

function MCMC_LinearRegress()

    % Length of series
    N = 1000;

    % MCMC assumptions
    Q = 5500; % Total number of MCMC simulations
    burn_in = 2500; % Burn-in period
    sigma_proposal = [0.1, 0.07, 0.1, 0.07]; % Proposal distribution standard deviations

    a_true = 1; % Intercept
    b_true = 2; % Slope for x1
    c_true = 3; % Slope for log(x1)
    s_true = 1.2; % Standard deviation of the error term

    % Normal distributed monte carlo
    x1 = sort(unifrnd(0, 10, N, 1)); % Uniformly distributed x1
    x2 = log(x1); % Log-transformed x1
    y = a_true + b_true * x1 + c_true * x2 + normrnd(0, s_true, N, 1); % Simulated y

    % OLS
    X = [ones(N, 1), x1, x2];
    est_ols = regress(y, X)';
    std_ols = std(y - X * est_ols');
    disp('------------------------------------------');
    disp('OLS');
    OLS_Estimates = [est_ols, std_ols]

    disp('global param');
    disp('True parameters:');
    True_Parameters = [a_true, b_true, c_true, s_true]

    % Init MCMC estimations
    est_mcmc = zeros(Q, 4);
    est_mcmc(1, :) = [1, 1, 1, 1]; %starting point

    % MCMC simulation
    for i = 2:Q
        est_current = est_mcmc(i - 1, :);
        for j = 1:4
            % Generate new proposal
            proposal = normrnd(est_current(j), sigma_proposal(j));
            est_current = MetropolisHastings(proposal, j, est_current, x1, x2, y);
        end
        est_mcmc(i, :) = est_current;
    end

    % Discard burn-in period
    est_mcmc = est_mcmc(burn_in + 1:end, :);

    % Compute MCMC estimates and standard errors
    mcmc_mean = mean(est_mcmc, 1);
    mcmc_std = std(est_mcmc, 1);

    disp('=========');
    disp('Estimated by MCMC method:');
    MCMC_Estimates = mcmc_mean
    MCMC_Standard_Errors = mcmc_std
    disp('==========');

    %hist
    figure;
    titles = {'1', '2', '3', '4'};
    for i = 1:4
        subplot(2, 2, i);
        hist(est_mcmc(:, i), 30);
        h = findobj(gca, 'Type', 'patch');
        set(h, 'FaceColor', [0.8, 0.8, 1]);
        title(sprintf('%s (\\mu = %.3f, \\sigma = %.3f)', ...
            titles{i}, mcmc_mean(i), mcmc_std(i)), 'FontSize', 14);
    end
end

function llh = loglikelihood(params, x1, x2, y)
    a = params(1);
    b = params(2);
    c = params(3);
    s = params(4);
    residuals = y - a - b * x1 - c * x2;
    llh = -0.5 * sum(log(s^2) + residuals.^2 / s^2);
end

function params = MetropolisHastings(proposal, index, params, x1, x2, y)
    params_new = params;
    params_new(index) = proposal;

    llh_old = loglikelihood(params, x1, x2, y);
    llh_new = loglikelihood(params_new, x1, x2, y);

    acceptance_prob = exp(llh_new - llh_old);
    if acceptance_prob > unifrnd(0, 1)
        params = params_new;
    end
end
