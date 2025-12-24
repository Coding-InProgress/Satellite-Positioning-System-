from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import PPO
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import os

from fastapi.middleware.cors import CORSMiddleware

# (20 satellites * 3 coordinates) + 1 active link count = 61
class SatelliteEnv(Env):
    def __init__(self, num_satellites=20):
        self.num_satellites = num_satellites
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

    def get_observation_space(self):
        # Match training environment: observations normalized between -1 and 1
        low = np.full(self.num_satellites * 3 + 1, -1.0, dtype=np.float32)
        high = np.full(self.num_satellites * 3 + 1, 1.0, dtype=np.float32)
        return Box(
            low=low,
            high=high,
            shape=(self.num_satellites * 3 + 1,),
            dtype=np.float32
        )

    def get_action_space(self):
        # Action space: 3D position changes for each satellite (dx, dy, dz)
        low = np.full(self.num_satellites * 3, -0.1, dtype=np.float32)
        high = np.full(self.num_satellites * 3, 0.1, dtype=np.float32)
        return Box(
            low=low,
            high=high,
            shape=(self.num_satellites * 3,),
            dtype=np.float32
        )

    def _get_obs(self, state):
        # The observation is the full state array
        return np.array(state, dtype=np.float32)

    def step(self, action):
        # Not used by the trained model — placeholder
        pass

    def reset(self, seed=None, options=None):
        # Not used by the trained model — placeholder
        pass


# FastAPI app
app = FastAPI()

# Add CORS middleware to allow frontend to connect
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the trained PPO model
model = None
MODEL_PATH = "ppo_satellite_model.zip"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {os.path.abspath(MODEL_PATH)}")
else:
    try:
        env = SatelliteEnv()
        model = PPO.load(MODEL_PATH, env=env)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")


# Request and Response models
class ActionRequest(BaseModel):
    positions: list[float]  # 20 satellites * 3 coordinates = 60 values
    active_links: int
    num_satellites: int


class ActionResponse(BaseModel):
    optimal_positions: list[list[float]]


@app.post("/get_optimal_action", response_model=ActionResponse)
def get_optimal_action(request: ActionRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="RL model is not loaded. Please check the server logs."
        )

    if request.num_satellites != 20:
        raise HTTPException(
            status_code=400,
            detail="The RL model requires exactly 20 satellites."
        )

    # Build state array
    state_array = np.array(
        request.positions + [request.active_links],
        dtype=np.float32
    )

    # Get the optimal action from the model
    action, _ = model.predict(state_array)

    # Reshape into vectors
    action_vector = action.reshape(-1, 3).tolist()

    # Update positions
    current_positions = np.array(request.positions).reshape(-1, 3)
    new_positions = current_positions + action.reshape(-1, 3)

    optimal_positions = new_positions.tolist()

    return {"optimal_positions": optimal_positions}
