% RL 6: Associability model 
% 
% Simulation & plotting script for the three-armed bandit (TAB) task 

% Reference: https://pubmed.ncbi.nlm.nih.gov/29313489/; https://pubmed.ncbi.nlm.nih.gov/21909088/
%% Set up
% Example block probabilities for a single block.
BlockProbs = [0.4937 0.8078 0.6343; % win probability
              0.5063 0.1922 0.3657];% loss probability

% Associability model parameters
% 1) learning rate (range between 0 and 1)
params.alpha = 0.5;
% 2) associability weight (range between 0 and 1)
params.eta = 0.5;
% 3) information bonus (any real value)
params.gamma = 0.01;
% 4) inverse temperature (range > 0)
params.beta = 6;

%% Run simulation

% For a decision task with N choices.
N_CHOICES = 3;
% Total number of trials (T) per block.
T = 16;

% Set up vectors with dimensions for both time and number of choices to keep track of
% components of choice value over time:
expected_value_w_info = zeros(N_CHOICES, T);    % overall expected value
expected_reward_value = zeros(N_CHOICES, T);    % value of expected reward
expected_info_value = zeros(N_CHOICES, T);      % value of expected information gain
last_chosen = zeros(N_CHOICES, T);              % trial number where an action was last taken 
associability = ones(N_CHOICES, T);             % associability values
    
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
    expected_info_value(:, t) = (t-last_chosen(:,t));
    expected_value_w_info(:, t) = expected_reward_value(:, t) + params.gamma*expected_info_value(:, t);
    
    % Compute action probabilities.
    P(:, t) = exp(params.beta * expected_value_w_info(:, t)) / sum(exp(params.beta * expected_value_w_info(1:N_CHOICES, t)));
    
    % Simulate actions.
    choice_at_t = find(rand < cumsum(P(:,t)),1);
    sim_choices(1,t) = choice_at_t;
    
    % Store the probability of chosen action on current trial.
    action_probabilities(t) = P(choice_at_t, t);
    
    % Copy previous values (to keep unchosen choices at
    % the same value from the previous timestep).
    expected_reward_value(:, t + 1) = expected_reward_value(:, t);
    last_chosen(:, t + 1) = last_chosen(:, t);
    last_chosen(choice_at_t, t + 1) = t;
    associability(:, t + 1) = associability(:, t);
    
    % Simulate rewards.
    outcome = find(rand < cumsum(BlockProbs(:, choice_at_t)), 1); % 1: win, 2: loss
    sim_rewards(1, t) =  -1 * (outcome -2); % 1: win, 0: loss
    
    % Compute prediction error.
    if sim_rewards(1, t) == 1
        prediction_error = sim_rewards(1, t) - expected_reward_value(choice_at_t, t);
    elseif sim_rewards(1, t) == 0
        reward_trans = -1; % transform loss value to -1
        prediction_error = reward_trans - expected_reward_value(choice_at_t, t);
    end
    
    % Update the associability values of the chosen action.
    associability(choice_at_t, t + 1) = (1 - params.eta) * associability(choice_at_t, t) + params.eta * abs(prediction_error);

    % Keep the associability values at a minimum of 0.05 and maximum of 1.
    associability(:, t + 1) = max(associability(:, t + 1), 0.05);
    associability(:, t + 1) = min(associability(:, t + 1), 1);
        
    % Update expected reward value of the chosen option.
    expected_reward_value(choice_at_t, t+1) = expected_reward_value(choice_at_t, t) + params.alpha*associability(choice_at_t, t)*prediction_error;
    
    % Store the value of prediction error. 
    prediction_error_sequence(t) = prediction_error;
    
    % Trims final value in expected reward value (due to an unnecessary 
    % value added at the end).
    expected_reward_value = expected_reward_value(:, 1:T);
    
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
