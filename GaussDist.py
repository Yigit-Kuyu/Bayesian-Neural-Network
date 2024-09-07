import gym
import numpy as np
import matplotlib.pyplot as plt



# Aim: 
# To model the uncertainty in the Q-values using a Gaussian distribution. 
# The mean represents the estimated Q-value, and the variance represents the uncertainty.


# Here, Q-values are modeled with a Gaussian distribution, where Q_mean represents the expected value
# and Q_var represents the uncertainty. The variance decreases as the agent becomes more certain about
# the value of the Q-function.



# Initialize the environment
env = gym.make('CartPole-v1')

# Hyperparameters
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
n_bins = 10  # Number of bins to discretize the state space

# Discretize the state space into bins
bins = [np.linspace(-4.8, 4.8, n_bins),  # Cart position
        np.linspace(-4, 4, n_bins),      # Cart velocity
        np.linspace(-0.418, 0.418, n_bins),  # Pole angle (rad)
        np.linspace(-4, 4, n_bins)]      # Pole velocity


# For a Gaussian (normal) distribution:
# The mean represents the most likely value of the quantity, 
# and the variance tells us how confident we are in this estimate.

# Initialize mean and variance of Q-values (Prior) (Gaussian distribution)
Q_mean = np.zeros((n_bins, n_bins, n_bins, n_bins, n_actions))
Q_var = np.ones((n_bins, n_bins, n_bins, n_bins, n_actions)) # captures the uncertainty in the Q-value estimate.

# Lists to store the evolution of Q-value mean and variance
q_mean_history = []
q_var_history = []


# Function to discretize a continuous state
def discretize_state(state_array):
    state_indices = []
    for i, s in enumerate(state_array):
        # Use np.digitize to assign each state component to a bin
        bin_idx = np.digitize(s, bins[i]) - 1  # Subtract 1 to get 0-based index
        # Ensure the indices are within the correct range (0 to n_bins-1)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        state_indices.append(int(bin_idx))  # Convert to int to avoid array issues
    return tuple(state_indices)


# Number of episodes
n_episodes = 10
# State-action pair to track
state_to_track = (5, 5, 5, 5)  # Middle state
action_to_track = 0

for episode in range(n_episodes):
    state = env.reset()
    state=state[0]
    state_discrete = discretize_state(state)
    done = False
    total_reward = 0
    
    #while not done:
    for ix in range(100):
        # Select action using the mean Q-values (could use UCB, LCB or Thompson Sampling)
        action = np.argmax(Q_mean[state_discrete])
        # sampled_q_values = np.random.normal(Q_mean[state_discrete], np.sqrt(Q_var[state_discrete])) # thompson sampling
        # action = np.argmax(sampled_q_values)
        
        # Take action and observe the outcome
        next_state, reward, done, _,_ = env.step(action)
        next_state_discrete = discretize_state(next_state)
        
        # Estimate the Q-value with uncertainty using Gaussian
        # TD difference between the observed reward plus the discounted future reward (gamma * np.max(Q_mean[next_state])) 
        # and the current Q-value estimate (Q_mean[state, action])
        td_target = reward + gamma * np.max(Q_mean[next_state_discrete])
        td_error = td_target - Q_mean[state_discrete + (action,)]
        
       
        # Bayesian update of mean and variance (Posterior)
        # Calculate the posterior variance using the Bayesian update rule for Gaussian distributions
        Q_var[state_discrete + (action,)] = 1 / (1 / Q_var[state_discrete + (action,)] + alpha)
        # Update the mean using the TD error and the posterior variance
        Q_mean[state_discrete + (action,)] += Q_var[state_discrete + (action,)] * td_error
        
        print(f"Episode: {episode}, Uncertainty (Variance): {Q_var[state_discrete, action]}")
        
        total_reward += reward
        state_discrete = next_state_discrete

    # Record Q-value mean and variance for the tracked state-action pair
    q_mean_history.append(Q_mean[state_to_track][action_to_track])
    q_var_history.append(Q_var[state_to_track][action_to_track])
    



# Plotting the Evolution of Q-value Mean and Variance for the Specific State-Action Pair
plt.figure(figsize=(12, 6))
plt.plot(np.arange(n_episodes), q_mean_history, label='Q-value Mean')
plt.fill_between(np.arange(n_episodes), 
                 np.array(q_mean_history) - np.sqrt(q_var_history),
                 np.array(q_mean_history) + np.sqrt(q_var_history),
                 alpha=0.2, label='Q-value Uncertainty')
plt.xlabel('Episode')
plt.ylabel('Q-value')
plt.title(f'Evolution of Q-value for State {state_to_track}, Action {action_to_track}')
plt.legend()
plt.show()

# Print final Q-value mean and variance for the tracked state-action pair
print(f"\nFinal Q-value for State {state_to_track}, Action {action_to_track}:")
print(f"Mean: {Q_mean[state_to_track][action_to_track]:.4f}")
print(f"Variance: {Q_var[state_to_track][action_to_track]:.4f}")
    


env.close()

