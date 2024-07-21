import gym
from gym import spaces
import numpy as np
import time
import logging
from game_control import GameControl

class PinballEnv(gym.Env):
    def __init__(self, window_title, templates_directory, screenshot_dir):
        super(PinballEnv, self).__init__()
        self.game_control = GameControl(window_title, templates_directory, screenshot_dir)
        self.reward_system = self.game_control.reward_system
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(7)
        self.last_snapshot_time = time.time()
        self.snapshot_interval = 1.0  # Time interval in seconds

    def reset(self):
        initial_state = self.game_control.reset_game()
        self.reward_system.reset()
        self.last_snapshot_time = time.time()
        return self._preprocess_state(initial_state)

    def step(self, action):
        current_time = time.time()
        if current_time - self.last_snapshot_time >= self.snapshot_interval:
            processed_frame, reward, done, info = self.game_control.perform_action(action)
            ball_count = self.reward_system.previous_ball_count
            score = self.reward_system.previous_score
            game_count = self.game_control.game_count
            cumulative_reward = self.game_control.cumulative_reward

            logging.info(f"Reward calculated in GameControl: {reward}")
            info.update({
                'screenshot': processed_frame,
                'ball_count': ball_count,
                'score': score,
                'game_count': game_count,
                'cumulative_reward': cumulative_reward
            })
            self.last_snapshot_time = current_time
            return self._preprocess_state(processed_frame), reward, done, info
        else:
            # If the interval has not passed, return zero reward and done as False
            return self._preprocess_state(np.zeros((480, 640, 3), dtype=np.uint8)), 0, False, {}

    def _preprocess_state(self, state):
        if state.shape != (480, 640, 3):
            state = np.transpose(state, (1, 0, 2))
        return state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
