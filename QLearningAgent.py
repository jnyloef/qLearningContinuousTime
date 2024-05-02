import numpy as np
from GeometricBrownianMotionEnv import *

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Q-learning agent.

        Args:
        - num_actions (int): Number of possible actions.
        - learning_rate (float): Learning rate for updating Q-values.
        - discount_factor (float): Discount factor for future rewards.
        - epsilon (float): Epsilon value for epsilon-greedy policy.
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q = {}

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy policy.

        Args:
        - state (float): The current state (stock price).

        Returns:
        - action (int): The chosen action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            if state not in self.Q:
                # Initialize Q-values for the state if not already done
                self.Q[state] = np.zeros(self.num_actions)
            return np.argmax(self.Q[state])

    def update_Q(self, state, action, reward, next_state):
        """
        Update Q-values based on the Q-learning update rule.

        Args:
        - state (float): The current state (stock price).
        - action (int): The action taken.
        - reward (float): The reward received for taking the action.
        - next_state (float): The next state (next stock price).
        """
        if state not in self.Q:
            # Initialize Q-values for the state if not already done
            self.Q[state] = np.zeros(self.num_actions)

        if next_state not in self.Q:
            # Initialize Q-values for the next state if not already done
            self.Q[next_state] = np.zeros(self.num_actions)

        # Q-learning update rule
        td_target = reward + self.discount_factor * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error

# Example usage
initial_price = 100
max_steps = 100
drift_rate = 0.05  # Annual drift rate
volatility = 0.2  # Annual volatility

env = GeometricBrownianMotionEnv(initial_price, max_steps, drift_rate, volatility)
agent = QLearningAgent(num_actions=3)  # 3 actions: Hold, Buy, Sell

num_episodes = 10

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update_Q(state, action, reward, next_state)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    print(f"Episode {episode + 1}: Final Price = {state}, Total Reward = {total_reward}")