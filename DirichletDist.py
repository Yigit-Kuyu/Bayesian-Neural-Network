import gym
import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt


# Aim: To use Dirichlet distribution as a prior over the transition probabilities in environments with discrete state spaces, such as FrozenLake. 
# The Dirichlet distribution is used to update and maintain beliefs over the possible transitions.



# Initialize the environment
env = gym.make('FrozenLake-v1')

# Hyperparameters
n_actions = env.action_space.n
n_states = env.observation_space.n
gamma = 0.99  # Discount factor
alpha = 0.1  # Learning rate

# Dirichlet distribution parameters for transition probabilities
# Initializing with 1s gives a uniform prior (all transitions equally likely)
dirichlet_params = np.ones((n_states, n_actions, n_states))

# Number of episodes
n_episodes = 1000

# Initialize Q-table
Q_table = np.zeros((n_states, n_actions))

# Initialize success probability and uncertainty tracking
Q_success_prob = np.zeros((n_states, n_actions))
Q_uncertainty = np.ones((n_states, n_actions))
successes = []

def get_state_index(state):
    if isinstance(state, tuple):
        return state[0]
    return state


# State-action pair to track
state_to_track = 0
action_to_track = 0
success_prob_history = []
for episode in range(n_episodes):
    state = env.reset()
    state_index = get_state_index(state)
    done = False
    

    while not done:
       # Sample transition probabilities from Dirichlet distribution
        sampled_transitions = dirichlet.rvs(dirichlet_params[state_index][0])
        
        # Select action based on current Q-table
        action = np.argmax(Q_table[state_index])
        
        # Take action and observe the outcome
        next_state, reward, done, _,_ = env.step(action)
        next_state_index = get_state_index(next_state)
        
        # Update the Dirichlet parameters with observed transition
        dirichlet_params[state_index, action, next_state_index] += 1
        
        # Update Q-value
        td_error = reward + gamma * np.max(Q_table[next_state_index]) - Q_table[state_index, action]
        Q_table[state_index, action] += alpha * td_error

         # Update success probability and uncertainty
        Q_success_prob[state_index, action] = dirichlet_params[state_index, action, next_state_index] / np.sum(dirichlet_params[state_index, action])
        Q_uncertainty[state_index, action] = 1 / (1 + np.sum(dirichlet_params[state_index, action]))
        
        state_index = next_state_index

    # Track success
    successes.append(reward)

    # Track success probability for specific state-action pair
    success_prob_history.append(Q_success_prob[state_to_track, action_to_track])


    if episode % 100 == 0:
        print(f"Episode {episode}: Success Probabilities and Uncertainty:")
        print("Success Probabilities:")
        print(Q_success_prob)
        print("Uncertainty:")
        print(Q_uncertainty)
        print(f"Average Success Rate in last 100 episodes: {np.mean(successes[-100:])}\n")

env.close()

print(f"\nFinal Success Probabilities after {n_episodes} episodes:")
print(Q_success_prob)
print(f"\nFinal Uncertainty after {n_episodes} episodes:")
print(Q_uncertainty)
print(f"\nOverall Average Success Rate: {np.mean(successes)}")

# Plotting the Evolution of Success Probability for a Specific State-Action Pair
plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, n_episodes), success_prob_history, label=f'P(success) for state {state_to_track}, action {action_to_track}')
plt.xlabel('Episode')
plt.ylabel('Success Probability')
plt.title(f'Evolution of Success Probability for State {state_to_track}, Action {action_to_track}')
plt.legend()
plt.show()

# Optional: Visualize the learned policy
def print_policy(Q_table):
    policy = np.argmax(Q_table, axis=1)
    policy_str = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    for i in range(4):
        for j in range(4):
            print(policy_str[policy[i*4 + j]][0], end=" ")
        print()

print("\nLearned Policy:")
print_policy(Q_table)

env.close()


