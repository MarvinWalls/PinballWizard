import gym
from gym import spaces
import numpy as np
from frame_processor import capture_screen, preprocess_screen
from reward_system import RewardSystem
from game_control import GameControl

class PinballEnv(gym.Env):
    """Custom Pinball Environment for 'The Arcade'."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PinballEnv, self).__init__()
        # Assuming GameControl handles game mechanics including initialization
        self.game_control = GameControl()
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # Example: No action, left flipper, right flipper, both, launch ball
        self.observation_space = spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8)

    def step(self, action):
        # Apply action, get next state, reward, and whether the game is done
        next_state, reward, done = self.game_control.perform_action(action)
        next_state = preprocess_screen(next_state)
        return next_state, reward, done, {}

    def reset(self):
        initial_state = self.game_control.reset_game()
        initial_state = preprocess_screen(initial_state)
        return initial_state
