import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(100)

class GeometricBrownianMotionEnv:
    def __init__(self, initial_price, final_time, max_steps, drift_rate, volatility):
        """
        Initialize the environment.
        
        Args:
        - initial_price (float): Initial price of the stock.
        - max_steps (int): Maximum number of steps (or trading days) in an episode.
        - drift_rate (float): Drift rate or the expected rate of return per unit time.
        - volatility (float): Volatility of the stock price, influencing the random changes in price at each step.
        """
        self.S0 = initial_price
        self.St = initial_price
        self.T = final_time
        self.max_steps = max_steps
        self.mu = drift_rate # mu
        self.sigma = volatility # sigma
        self.dt = final_time/max_steps  # Time increment
        self.t = np.linspace(0, final_time, max_steps)
        # self.Wt = self.brownian_motion()
        self.steps_elapsed = 0

    def brownian_motion(self):
        N=20000
        t=self.t.reshape(1,-1)
        k=np.arange(N).reshape(-1,1)
        Z=np.random.randn(N).reshape(-1,1)
        Wt=np.sqrt(2*self.T)/np.pi*np.sum( np.sin((k+0.5) @ t * np.pi/self.T)/(k+0.5)*Z , axis=0)
        #plt.figure()
        #plt.plot(t[0],x, linewidth = 0.7)
        #plt.grid()
        #plt.show()
        return Wt

    def reset(self):
        """Reset the environment to its initial state."""
        self.St = self.S0
        self.steps_elapsed = 0
        return self.St

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
        - action (int): Action taken by the agent (not used in this environment).

        Returns:
        - state (float): The current state (stock price).
        - reward (float): The reward for the action taken (always 0 in this environment).
        - done (bool): Whether the episode is done.
        - info (dict): Additional information (not used in this environment).
        """
        
        # Generate random Brownian motion increment
        dWt = np.sqrt(self.dt) * np.random.randn() # should also work

        # Update price based on geometric Brownian motion formula (Euler forward)
        drift_component = self.mu * self.St * self.dt
        diffusion_component = self.sigma * self.St * dWt
        dSt = drift_component + diffusion_component
        self.St += dSt

        # Define reward based on action (for example, holding)
        reward = 0

        # Check if episode is done
        self.steps_elapsed += 1
        done = self.steps_elapsed >= self.max_steps

        return self.St, reward, done, {}
    

# UNIT TEST 
if __name__ == "__main__":
    # Example usage
    initial_price = 100
    final_time = 10
    max_steps = 1000

    drift_rate = 1  # Annual drift rate
    volatility = 1  # Annual volatility

    env = GeometricBrownianMotionEnv(initial_price, final_time, max_steps, drift_rate, volatility)
    S = np.zeros(max_steps)
    W = np.zeros(max_steps)
    S[0] = env.S0
    for i in range(max_steps-1):
        Stpdt, reward, done, _ = env.step(action = None)
        S[i+1] = Stpdt

    plt.figure()
    plt.plot(env.t, S)
    #plt.plot(env.t, env.S0*np.exp((env.mu - 0.5*env.sigma**2)*env.t + env.sigma*env.Wt))
    plt.grid()
    plt.show()