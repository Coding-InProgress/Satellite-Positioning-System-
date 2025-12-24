import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import SatelliteEnv

# Create the environment
# n_envs=1 creates a single vectorized environment, which is suitable for this task.
env = make_vec_env(SatelliteEnv, n_envs=1)

# Initialize the PPO (Proximal Policy Optimization) agent
# 'MlpPolicy' is a Multi-Layer Perceptron neural network policy, which is a good choice for this type of environment.
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent for 100,000 timesteps. 
# This number can be adjusted (higher for better performance, lower for faster training).
print("Starting training...")
model.learn(total_timesteps=100000)
print("Training complete.")

# Save the trained model to a file.
# The model is saved as a .zip archive, which can be loaded later by the API.
model.save("ppo_satellite_model")
print("Model saved as ppo_satellite_model.zip")
