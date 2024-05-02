import numpy as np
from GridWorld import GridWorld

class QLearning:
    def __init__(self, num_states, num_actions, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        next_state, reward, done = self.env.step(action)
        return next_state, reward, done

# Example usage
grid_size = 5
env = GridWorld(grid_size)
num_states = grid_size * grid_size
num_actions = 4

ql = QLearning(num_states, num_actions, env)

num_episodes = 1000
max_steps_per_episode = 100

for episode in range(num_episodes):
    state = ql.reset()
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = ql.choose_action(state)
        next_state, reward, done = ql.step(action)
        ql.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    print("Episode:", episode, "Total Reward:", total_reward)

print("Q-table:", ql.q_table)
