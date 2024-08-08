import gym
from gym import spaces
import numpy as np
import logging
from game_control import GameControl
import cv2

class PinballEnv(gym.Env):
    def __init__(self, window_title, templates_directory, screenshot_dir):
        super(PinballEnv, self).__init__()
        self.game_control = GameControl(window_title, templates_directory, screenshot_dir)
        self.reward_system = self.game_control.reward_system
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(7)
        logging.info("PinballEnv initialized")

    def _capture_state(self):
        frame, _ = self.game_control.capture_screen()
        return self._preprocess_state(frame)

    def reset(self):
        initial_state = self.game_control.reset_game()
        self.reward_system.reset()
        return self._preprocess_state(initial_state)

    def step(self, action):
        processed_frame, reward, done, info = self.game_control.perform_action(action)
        ball_count = self.reward_system.previous_ball_count
        score = self.reward_system.previous_score
        logging.info(f"Reward calculated in GameControl: {reward}")
        info.update({
            'screenshot': processed_frame,
            'ball_count': ball_count,
            'score': score,
            'reward': reward
        })
        return self._preprocess_state(processed_frame), reward, done, info

    def _preprocess_state(self, state):
        if state.shape != (240, 320, 3):
            state = np.stack([state] * 3, axis=-1) if len(state.shape) == 2 else state
            state = cv2.resize(state, (320, 240))
        return state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        logging.info("PinballEnv closed")
