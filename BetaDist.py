import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


# Aim: 
# To model the success or failure of actions in a binary outcome environment. 
# This is particularly useful for environments where the reward is binary (e.g., 0 or 1).



# In this setup, The Beta distribution parameters (alpha and beta) for each state-action pair represent our accumulated knowledge about successes and failures.
# The probability is updated based on the observed reward (binary outcome), which is appropriate for
# environments like FrozenLake where rewards are often binary (success/failure).


# Initialize the environment
env = gym.make('FrozenLake-v1')

# Hyperparameters
n_actions = env.action_space.n
n_states = env.observation_space.n
gamma = 0.99  # Discount factor
alpha_prior = 1.0  # Initial Beta distribution parameter (prior)
beta_prior = 1.0   # Initial Beta distribution parameter (prior)

# Initialize Beta distribution parameters for success probabilities
Q_alpha = np.full((n_states, n_actions), alpha_prior)  # Alpha parameter of Beta distribution (prior)
Q_beta = np.full((n_states, n_actions), beta_prior)    # Beta parameter  of Beta distribution (prior)


# Number of episodes
n_episodes = 1000
successes = []

for episode in range(n_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Extract the integer state value
        state_index = state[0] if isinstance(state, tuple) else state

         # Thompson sampling for action selection (beta distribution)
        sampled_probs = np.random.beta(Q_alpha[state_index], Q_beta[state_index])
        action = np.argmax(sampled_probs)
        
        
        # Take action and observe the outcome
        next_state, reward, done,_,_= env.step(action)
        
       # Update the Beta distribution parameters based on observed reward
        if reward == 1:
            Q_alpha[state_index, action] += 1  # Increment alpha for success (posterior update)
        else:
            Q_beta[state_index, action] += 1   # Increment beta for failure (posterior update)
        
        
        state = next_state
    
    successes.append(total_reward)
    if episode % 100 == 0:
        print(f"Episode {episode}: Average Success Rate in last 100 episodes: {np.mean(successes[-100:])}")




# Calculate and print final success probabilities and uncertainties
success_probs = Q_alpha / (Q_alpha + Q_beta)
uncertainties = np.sqrt((Q_alpha * Q_beta) / ((Q_alpha + Q_beta)**2 * (Q_alpha + Q_beta + 1)))

print("\nFinal Success Probabilities:")
print(success_probs)
print("\nFinal Uncertainties:")
print(uncertainties)



plt.figure(figsize=(10, 5))
plt.plot(np.convolve(successes, np.ones(100)/100, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Average Success Rate (over last 100 episodes)')
plt.title('Learning Curve')
plt.show()

# Plot Beta distribution for a specific state-action pair
state_to_plot = 0
action_to_plot = 0
x = np.linspace(0, 1, 100)
y = beta.pdf(x, Q_alpha[state_to_plot, action_to_plot], Q_beta[state_to_plot, action_to_plot])

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.xlabel('Success Probability')
plt.ylabel('Density')
plt.title(f'Beta Distribution for State {state_to_plot}, Action {action_to_plot}')
plt.show()


