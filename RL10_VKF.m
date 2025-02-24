% RL 10: volatile Kalman filter model 
% 
% Simulation & plotting script for the three-armed bandit (TAB) task

% Reference: https://pubmed.ncbi.nlm.nih.gov/32609755/
% Adapted from https://github.com/payampiray/VKF/blob/master/vkf_bin.m
%% Set up
% Example block probabilities for a single block.
BlockProbs = [0.4937 0.8078 0.6343; % win probability
              0.5063 0.1922 0.3657];% loss probability

% VKF model parameters
% 1) initial volatility (range > 0)
params.v0 = 5;
% 2) expected noise (range > 0)
params.omega = 1;
% 3) volatility update rate (range between 0 and 1)
params.lambda = 0.2;
% 4) information bonus (any real value)
params.gamma = 1;
% 5) inverse temperature (range > 0)
params.beta = 6;

%% Run simulation

% For a decision task with N choices.
N_CHOICES = 3;
% Total number of trials (T) per block.
T = 16;

% Set up vectors with dimensions for both time and number of choices to keep track of
% components of choice value over time:
expected_value_w_info = zeros(N_CHOICES, T);    % overall expected value
expected_info_value = zeros(N_CHOICES, T);      % value of expected information gain
mean = zeros(N_CHOICES, T);                     % Mean of the reward probability of each option
variance = params.omega*ones(N_CHOICES, T);     % Variance of the reward probability of each option
sigmoid_mean = zeros(N_CHOICES, T);             % Sigmoid transformed mean
kalman_gain = zeros(N_CHOICES, T);              % Kalman gain
learning_rate = zeros(N_CHOICES, T);            % Learning rate
volatility = params.v0*ones(N_CHOICES, T);      % Volatility
auto_cov = zeros(N_CHOICES, T);                 % Covariance

% Represents the probability distribution (P) over choices on each trial.
P = zeros(N_CHOICES, T);

% Empty vectors to store stimulated choices, rewards, action probabilities, and
% prediction error.
sim_choices = zeros(1, T);
sim_rewards = zeros(1, T);
action_probabilities = zeros(1, T);
prediction_error_sequence = zeros(1,T);

% For each trial (timestep)...
for t = 1:T
    
    % Compute expected value for each option.
    expected_info_value(:, t) = sqrt(variance(:,t));
    expected_value_w_info(:, t) = mean(:,t) + params.gamma*expected_info_value(:, t);
    
    % Compute action probabilities.
    P(:, t) = exp(params.beta * expected_value_w_info(:, t)) / sum(exp(params.beta * expected_value_w_info(1:N_CHOICES, t)));
    
    % Simulate actions.
    choice_at_t = find(rand < cumsum(P(:,t)),1);
    sim_choices(1,t) = choice_at_t;
    
    % Store the probability of chosen action on current trial.
    action_probabilities(t) = P(choice_at_t, t);
    
    % Copy previous values (to keep unchosen choices at
    % the same value from the previous timestep).
    mean(:, t+1) = mean(:, t);
    variance(:,t+1) = variance(:, t);
    volatility(:,t+1) = volatility(:, t);
    kalman_gain(:, t+1) = kalman_gain(:, t);
    learning_rate(:, t+1) = learning_rate(:, t);
    auto_cov(:, t+1) = auto_cov(:, t);
    
    % Update kalman_gain of the chosen option.
    kalman_gain(choice_at_t, t+1) = (variance(choice_at_t, t) + volatility(choice_at_t, t))'./(variance(choice_at_t, t) + volatility(choice_at_t, t) + params.omega)';
    
    % Update learning rate of the chosen option.
    learning_rate(choice_at_t, t+1) = sqrt(variance(choice_at_t, t) + volatility(choice_at_t, t));
    
    % Simulate rewards.
    outcome = find(rand < cumsum(BlockProbs(:, choice_at_t)), 1); % 1: win, 2: loss
    sim_rewards(1, t) =  -1 * (outcome -2); % 1: win, 0: loss
    
    % Compute prediction error.
    sigmoid_mean(:, t) = 1./(1+exp(-(mean(:,t)')));
    prediction_error = sim_rewards(1, t) - sigmoid_mean(choice_at_t, t);
    
    % Update mean of the chosen option.
    mean(choice_at_t, t+1) = mean(choice_at_t, t) + learning_rate(choice_at_t, t+1)*prediction_error;

    % Update variance of the chosen option.
    variance(choice_at_t, t+1) = (1 - kalman_gain(choice_at_t, t+1))*(variance(choice_at_t, t) + volatility(choice_at_t, t));

    % Update covariance of the chosen option.
    auto_cov(choice_at_t, t+1) = (1 -  kalman_gain(choice_at_t, t+1)).*variance(choice_at_t, t);
    
    % Update volatility.
    volatility(choice_at_t, t+1) = volatility(choice_at_t, t) + params.lambda*((mean(choice_at_t, t+1) - mean(choice_at_t, t)).^2 + variance(choice_at_t, t+1) + variance(choice_at_t, t) - 2*auto_cov(choice_at_t, t+1) - volatility(choice_at_t, t));

    % Store the value of prediction error.  
    prediction_error_sequence(t) = prediction_error;
    
    % Trims final value (due to an unnecessary value added at the end).
    sigmoid_mean = sigmoid_mean(:, 1:T);
    mean = mean(:, 1:T);
    variance = variance(:, 1:T);
    volatility = volatility(:, 1:T);
    kalman_gain = kalman_gain(:, 1:T);
    learning_rate = learning_rate(:, 1:T);
    auto_cov = auto_cov(:, 1:T);
    
end

%% Plotting
% Plot the choices(blue), outcomes(green/red), and action probabilties (gray scale; darker = higher probability) 
% of all options across trials (timestep).
col = {[0, 0.4470, 0.7410], ...       % blue
       [0.4660, 0.6740, 0.1880], ...  % green
       [0.9350, 0.1780, 0.2840], ...  % red
      };                    
cols = [0:1/32:1; 0:1/32:1; 0:1/32:1]';
MarkerSize = 10;

figure(1)
for i = 1:T
    if sim_rewards(1, i) == 1
        plot(i,4,'o','MarkerSize',MarkerSize,"MarkerEdgeColor", 'black','MarkerFaceColor',col{2})
        hold on
    else
        plot(i,4,'o','MarkerSize',MarkerSize,"MarkerEdgeColor", 'black','MarkerFaceColor',col{3})
        hold on
    end
end

imagesc([1 - P]); colormap(cols) , hold on
plot([sim_choices],'o','MarkerSize',MarkerSize,"MarkerEdgeColor", col{1},'MarkerFaceColor',col{1})
xlabel("trial number (timestep)")
title('Action Probabilities and Chosen Actions')
xlim([-1,T+1])
set(gca, 'XTick', [0:T]),
set(gca, 'YTick', [1:4]), set(gca, 'YTickLabel', {'Option 1','Option 2','Option 3' 'Outcomes' })
