class GridWorld:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.state = (0, 0)  # Starting state at the top-left corner
        self.goal_state = (grid_size - 1, grid_size - 1)  # Goal state at the bottom-right corner

    def reset(self):
        self.state = (0, 0)  # Reset to the starting state
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Move right
            x += 1
        elif action == 1:  # Move down
            y += 1
        elif action == 2:  # Move left
            x -= 1
        elif action == 3:  # Move up
            y -= 1

        # Keep the agent within the grid boundaries
        x = max(0, min(x, self.grid_size - 1))
        y = max(0, min(y, self.grid_size - 1))

        self.state = (x, y)

        # Determine reward and if episode is done
        reward = 0
        done = False
        if self.state == self.goal_state:
            reward = 1
            done = True

        return self.state, reward, done