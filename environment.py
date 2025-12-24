import numpy as np
import random
from gymnasium import Env
from gymnasium.spaces import Box

# Define the satellite environment for reinforcement learning.
# This class simulates the state, actions, and rewards for a constellation of satellites.
class SatelliteEnv(Env):
    """
    Custom Gymnasium environment for a satellite constellation.
    The goal is to maximize the number of active communication links.
    """
    def __init__(self, num_satellites=20):
        super(SatelliteEnv, self).__init__()
        self.num_satellites = num_satellites
        # Calculate the maximum possible links in the constellation
        self.max_links = num_satellites * (num_satellites - 1) / 2  

        # Define the observation space (the state).
        # It includes the (x, y, z) position of each satellite and the normalized link count.
        self.observation_space = Box(low=-1.0, high=1.0, shape=(num_satellites * 3 + 1,), dtype=np.float32)
        
        # Define the action space (the possible moves).
        # This is a continuous space representing small adjustments to each satellite's position.
        self.action_space = Box(low=-0.1, high=0.1, shape=(num_satellites * 3,), dtype=np.float32)

        self.current_positions = None
        self.current_links = 0
        self.state = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        Satellites are placed in random positions.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Initialize satellites in random positions within the [-1, 1] bounds
        self.current_positions = np.random.uniform(low=-1.0, high=1.0, size=(self.num_satellites, 3))
        # Calculate the initial number of active links
        self.current_links = self._calculate_links(self.current_positions)
        
        # Construct the state vector: flattened positions + normalized link count
        self.state = np.append(self.current_positions.flatten(), self.current_links / self.max_links)
        info = {}
        return self.state, info

    def step(self, action):
        """
        Takes an action and updates the environment.
        The agent's action (movement) is applied, and a reward is calculated.
        """
        # Reshape the action vector to apply movement to each satellite individually
        movement = action.reshape(self.num_satellites, 3)
        self.current_positions += movement
        
        # Clip positions to ensure they stay within the observation space bounds
        self.current_positions = np.clip(self.current_positions, -1.0, 1.0)
        
        # Calculate the number of links after the movement
        new_links = self._calculate_links(self.current_positions)
        
        # The reward is the change in the number of links
        reward = new_links - self.current_links
        self.current_links = new_links
        
        # The episode is never terminated in this continuous task, but this can be adjusted.
        terminated = False
        truncated = False
        
        # Update the state with the new positions and link count
        self.state = np.append(self.current_positions.flatten(), self.current_links / self.max_links)
        info = {}
        return self.state, reward, terminated, truncated, info

    def _calculate_links(self, positions):
        """
        A simple helper method to count "active" links based on a distance threshold.
        """
        link_count = 0
        # A link is considered active if two satellites are within this distance
        distance_threshold = 0.5  
        for i in range(self.num_satellites):
            for j in range(i + 1, self.num_satellites):
                # Calculate the Euclidean distance between two satellites
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < distance_threshold:
                    link_count += 1
        return link_count