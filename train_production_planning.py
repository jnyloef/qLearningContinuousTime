import numpy as np
import matplotlib.pyplot as plt
from ProductionPlanningEnv import *
from ContTimeQLearningAgent import *

def train_qlearning_agent(agent, env, num_episodes, update_mode):
    episodes = [None]*num_episodes
    for episode in range(num_episodes):
        trajectory = {'times':[None]*env.max_steps,
                      'states':[None]*env.max_steps,
                      'actions':[None]*env.max_steps,
                      'rewards':[None]*env.max_steps,
                      'next_states':[None]*env.max_steps}
        Xt = env.reset()
        for step in range(env.max_steps):
            t = step * env.dt # time
            at = agent.choose_action(t, Xt)
            Xtpdt, rt, done = env.step(at) # states are updated in this function
            if update_mode == 'on':
                agent.update(t, Xt, at, rt, Xtpdt)
            trajectory['times'][step] = t
            trajectory['states'][step] = Xt
            trajectory['actions'][step] = at
            trajectory['rewards'][step] = rt
            trajectory['next_states'][step] = Xtpdt
            Xt = Xtpdt # go to next state
        
        assert done == True
        if update_mode == 'off':
            agent.update(trajectory['times'], trajectory['states'], trajectory['actions'], trajectory['rewards'], trajectory['next_states'])

        episodes[episode] = trajectory
        print(f"Episode {episode + 1}: Final Inventory = {trajectory['states'][-1]}, Non-discounted Total Reward = {env.dt*np.sum(trajectory['rewards']) + env.h_func(trajectory['states'][-1])}")
    
    return episodes

if __name__ == '__main__':
    initial_inventory = 0
    max_steps = 10
    demand = 1  # Annual drift rate
    volatility = 0.2  # Annual volatility
    B = 10 # Parameter for h function
    final_time = 10
    num_episodes = 10
    update_mode  = 'off'
    beta = 0.1
    gamma = 1
    learning_rate = 0.1

    env = ProductionPlanningEnv(final_time, max_steps, initial_inventory, demand, volatility, B)
    agent = ContTimeQLearningAgent(final_time, max_steps, beta, gamma, env.h_func, learning_rate)
    episodes = train_qlearning_agent(agent, env, num_episodes, update_mode)