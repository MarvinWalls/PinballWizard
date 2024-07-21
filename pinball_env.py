import gym
from gym import spaces
import numpy as np
from game_control import GameControl
from reward_system import RewardSystem

class PinballEnv(gym.Env):
    def __init__(self, window_title, templates_directory, screenshot_dir):
        super(PinballEnv, self).__init__()
        self.game_control = GameControl(window_title, templates_directory, screenshot_dir)
        self.reward_system = self.game_control.reward_system  # Ensure reward_system is initialized
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(7)

    def reset(self):
        initial_state = self.game_control.reset_game()
        self.reward_system.reset()  # Reset reward system on environment reset
        return self._preprocess_state(initial_state)

    def step(self, action):
        processed_frame, reward, done = self.game_control.perform_action(action)
        info = {
            'ball_count': self.reward_system.current_ball_count,
            'score': self.reward_system.current_score
        }
        return self._preprocess_state(processed_frame), reward, done, info

    def _preprocess_state(self, state):
        if state.shape != (480, 640, 3):
            state = np.transpose(state, (1, 0, 2))
        return state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
