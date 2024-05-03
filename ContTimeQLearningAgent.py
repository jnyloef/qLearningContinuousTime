import numpy as np
from GeometricBrownianMotionEnv import *
from ProductionPlanningEnv import *
from NeuralNetwork import *

class ContTimeQLearningAgent:
    def __init__(self, final_time, max_steps, beta, gamma, h_func, learning_rate):
        """
        Initialize the continuous time q-learning agent.

        Args:
        - num_actions (int): Number of possible actions.
        - learning_rate (float): Learning rate for updating Q-values.
        - discount_factor (float): Discount factor for future rewards.
        - epsilon (float): Epsilon value for epsilon-greedy policy.
        """
        self.T = final_time
        self.max_steps = max_steps
        self.dt = final_time/max_steps  # Time increment
        self.t_list = np.linspace(0, final_time, max_steps+1)
        self.beta = beta
        self.gamma = gamma
        self.h_func = h_func
        self.learning_rate = learning_rate
        layers = [2, 100, 50, 10, 5, 1] # (t,x) \to R
        self.NN_psi1 = NeuralNetwork(layers) # variable psi1
        self.NN_psi2 = NeuralNetwork(layers) # variable psi2
        self.NN_theta = NeuralNetwork(layers) # variable theta

    def J_theta(self, t, x):
        X = np.array([t,x]).reshape(-1,1)
        return (t/self.T)*self.h_func(x) + (1 - t/self.T)*self.NN_theta.eval(X)
    
    def q_psi(self, t, x, a):
        X = np.array([t,x]).reshape(-1,1)
        NN_ps1 = self.NN_psi1.eval(X)
        NN_ps2 = self.NN_psi2.eval(X)
        return -0.5 * np.exp(NN_ps2)*(a - NN_ps1)**2 + 0.5*self.gamma * (NN_ps2 - np.log(2*np.pi*self.gamma))

    def choose_action(self, time, state):
        """
        Choose an action from the Gibbs Measure associated with the current q-function approximation.

        Args:
        - Xt (float): The current state.
        - t (float): The current time.

        Returns:
        - action (int): The chosen action from the Gibbs measure for current q-function, not optimal one, hence off-policy learning.
        In the case of a quadratic q-function, the Gibbs measure is a normal distribution.
        """
        time_state_pair = np.array([time, state]).reshape(-1,1) # column vector
        mean = self.NN_psi1.eval(time_state_pair)
        std_dev = np.sqrt( self.gamma / np.exp(self.NN_psi2.eval(time_state_pair)) )
        return mean + std_dev * np.random.randn()

    def update(self, times, states, actions, rewards, next_states):
        """
        Update Q-values based on the Q-learning update rule in both the online and offline setting.

        Args:
        - times (float or list): The times 0 = t_0 < t_1 < ... < t_{K-1}. For online algorithm, K = 1.
        - states (float or list): The states corresponding to times 0 = t_0 < t_1 < ... < t_{K-1}..
        - actions (float or list): The actions taken corresponding to times 0 = t_0 < t_1 < ... < t_{K-1}..
        - rewards (float or list): The rewards received corresponding to times 0 = t_0 < t_1 < ... < t_{K-1}..
        - next_states (float or list): The states corresponding to times t_1 < t_2 < ... < t_{K} = T.
        - episode (int): The index of the current episode.
        """
        if isinstance(times, float):
            times = [times]; states = [states]; actions = [actions]; rewards = [rewards], next_states = [next_states]
        K = len(times)
        # compute total over times update-increments for the weights and biases in each neural network at each layer.
        NN = [self.NN_psi1, self.NN_psi2, self.NN_theta]
        update_increments_weights = {i:[0.0]*(NN[i].num_layers - 1) for i in range(len(NN))}
        update_increments_biases = {i:[0.0]*(NN[i].num_layers - 1) for i in range(len(NN))}
        for j in range(K):
            time_state_pair = np.array([times[j], states[j]]).reshape(-1,1)
            delta = self.J_theta(times[j] + self.dt, next_states[j]) - self.J_theta(times[j], states[j]) + self.dt * ( reward - self.q_psi(times[j], states[j], actions[j]) - self.beta*self.J_theta(times[j], states[j]) )
            factors = [np.exp(self.NN_psi2.eval(time_state_pair))*(actions[j] - self.NN_psi2.eval(time_state_pair)),
                    0.5*(self.gamma - np.exp(self.NN_psi2.eval(time_state_pair))*(actions[j] - self.NN_psi1.eval(time_state_pair))**2),
                    1 - times[j]/self.T]
            for i in range(len(NN)):
                weight_grads_X, bias_grads_X = NN[i].gradient(time_state_pair)
                for l in range(NN[i].num_layers - 1):
                    update_increments_weights[i][l] += self.learning_rate * factors[i] * weight_grads_X * delta
                    update_increments_biases[i][l] += self.learning_rate * factors[i] * bias_grads_X * delta

        # Update weights and biases in the neural network by adding the update-incements. Has to be done in this order to not alter the neural network when computing the increments.
        for i in range(len(NN)):
            for l in range(NN[i].num_layers - 1):
                NN[i].weights[l] += update_increments_weights[i][l]
                NN[i].biases[l] += update_increments_biases[i][l]

