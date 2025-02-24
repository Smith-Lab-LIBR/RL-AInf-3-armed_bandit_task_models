% AInf 8: Dynamic forgetting + 2 learning rates (LR) model 
% 
% Simulation & plotting script for the three-armed bandit (TAB) task 

% Reference: https://pubmed.ncbi.nlm.nih.gov/30608922/
% Other resources: active inference tutorial scripts https://github.com/rssmith33/Active-Inference-Tutorial-Scripts
%% To use this code, make sure to add the path to the SPM package here:
addpath("XXX")

%% Set up
% Example block probabilities for a single block.
BlockProbs = [0.4937 0.8078 0.6343;  % win probability
              0.5063 0.1922 0.3657]; % loss probability

% AInf dynamic forgetting + 2LR model parameters
% 1) learning rate for wins (range > 0)
params.alpha_win = 0.5;
% 1) learning rate for losses (range > 0)
params.alpha_loss = 0.8;
% 2) baseline forgetting rate (range between 0 and 1)
params.psi_0 = 0.25;
% 3) forgetting update rate (range between 0 and 1)
params.psi_r = 0.10;
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
% components of choice value (negative expected free energy) over time:
expected_value_w_info = zeros(N_CHOICES, T);     % overall expected value (negative expected free energy)
expected_info_value = zeros(N_CHOICES, T);       % value of expected information gain
expected_reward_value = zeros(N_CHOICES, T);     % value of expected reward

% Represents the probability distribution (P) over choices on each trial.
P = zeros(N_CHOICES, T);

% Empty vectors to store stimulated choices, rewards, action probabilities,
% state-action prediction error (SAPE), and trial-by-trial forgetting rate.
sim_choices = zeros(1, T);
sim_rewards = zeros(1, T);
action_probabilities = zeros(1, T);
SAPE = zeros(1,T);
psi = zeros(1,T);

% v_0: Initial values for concentration parameters in a Dirichlet distribution (v), which encodes 
% a prior over the likelihood matrix, V, which represents beliefs about reward probabilities. Columns  
% correspond to choice states, rows correspond to observing wins (row 1) or losses (row 2).  
% Each element can take any positive value, encoding initial confidence in each reward probability.  
% Each column is normalized to create the expected reward probabilities on each trial in V.
% Here, we initialize baseline confidence at 0.5. 
v_0 = [0.5 0.5 0.5;
       0.5 0.5 0.5];

% B_pi: A matrix representing deterministic transitions to each bandit state, depending on choice.
% Each column represents a choice. Each row represents a bandit state.
B_pi = [1 0 0;
        0 1 0;
        0 0 1];

% R vector: encodes reward value for [win loss] observations. Often referred to as a "preference distribution"
% in Active Inference.
R = spm_softmax([4 0]');

% Outcome vector: used to encode a win or loss on each trial and update the v matrix during learning.
outcome_vector = zeros(2,T);

% For each trial (timestep)...
for t = 1:T
    
    % Normalize v to create likelihood matrix V (representing expected
    % reward probabilities).
    if t == 1
        V{t} = spm_norm(v_0);
        v{t} = v_0;
    else
        V{t} = spm_norm(v{t});
    end
      
    % Compute expected information gain (based on sum of current
    % concentration parameter counts).
    v_sums{t} = [sum(v{t}(:,1)) sum(v{t}(:,2)) sum(v{t}(:,3));
                 sum(v{t}(:,1)) sum(v{t}(:,2)) sum(v{t}(:,3))];
              
    info_gain = .5*((v{t}.^-1)-(v_sums{t}.^-1));
    
    % Compute expected value (negative expected free energy) for each
    % option (pol).
    for pol = 1:3 
        % expected information gain
        expected_info_value(pol,t) = dot(V{t}*B_pi(:,pol),info_gain*B_pi(:,pol));
        % expected reward
        expected_reward_value(pol,t) = dot(V{t}*B_pi(:,pol),log(R));
        % overall expected value
        expected_value_w_info(pol,t) = expected_reward_value(pol,t)+ params.gamma*expected_info_value(pol,t);
    end
    
    % Compute action probabilities.
    P(:,t) = exp(params.beta*expected_value_w_info(:,t))/sum(exp(params.beta*expected_value_w_info(:,t)));
    
    % Compute state-action-prediction error (SAPE), defined as a KL divergence
    % between the probability distribution over actions on the current and
    % previous trial.
    if t >= 2
        SAPE(1,t) = sum(P(:,t)'.*(log(P(:,t)')-log(P(:,t-1)')));
    end
    
    % Simulate actions.
    choice_at_t = find(rand < cumsum(P(:,t)),1);
    sim_choices(1,t) = choice_at_t;
    
    % Store the probability of chosen action on current trial.
    action_probabilities(t) = P(choice_at_t, t);

    % Simulate rewards.
    sim_rewards(1,t) = find(rand < cumsum(BlockProbs(:,choice_at_t)),1); % 1: win, 2: loss
    outcome_vector(sim_rewards(1,t),t) = 1;
    
    % Compute forgetting rate for current trial (a baseline forgetting 
    % rate and the product between the forgetting update rate and SAPE).
    psi(1,t) = params.psi_0 + params.psi_r*SAPE(1,t);
    % Upper bound on forgetting rate (psi <= 1)
    psi(1,t) = min(psi(1,t), 1);
    
    % Forgetting (scale down concentration parameter counts in Dirichlet
    % distribution v).
    % v_0 acts as a lower bound on forgetting.
    v{t+1} = (v{t} - v_0)*(1 - psi(t)) + v_0;
    
    % Learning (update concentration parameter counts in Dirichlet distribution
    % v).
    if sim_rewards(1,t) == 1
        v{t+1} = v{t+1} + params.alpha_win*(B_pi(:,choice_at_t)*outcome_vector(:,t)')';
    elseif sim_rewards(1,t) == 2
        v{t+1} = v{t+1} + params.alpha_loss*(B_pi(:,choice_at_t)*outcome_vector(:,t)')';
    end
 
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

function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);
end