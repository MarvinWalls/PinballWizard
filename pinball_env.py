import gym
from gym import spaces
import numpy as np
import time
import threading
from game_control import GameControl
import logging

class PinballEnv(gym.Env):
    def __init__(self, window_title, templates_directory, screenshot_dir):
        super(PinballEnv, self).__init__()
        self.game_control = GameControl(window_title, templates_directory, screenshot_dir)
        self.reward_system = self.game_control.reward_system
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(7)
        self.last_snapshot_time = time.time()
        self.snapshot_interval = .25  # Time interval in seconds (4 screenshots per second)
        self.running = True

        # Start the screenshot capturing thread
        self.screenshot_thread = threading.Thread(target=self.capture_screenshots)
        self.screenshot_thread.start()

    def capture_screenshots(self):
        while self.running:
            current_time = time.time()
            if current_time - self.last_snapshot_time >= self.snapshot_interval:
                state = self._capture_state()
                reward = self.reward_system.calculate_reward(self.game_control.get_current_score(state), self.game_control.get_current_ball_count(state))
                info = {
                    'screenshot': state,
                    'reward': reward,
                    'ball_count': self.reward_system.previous_ball_count,
                    'score': self.reward_system.previous_score,
                    'game_count': self.game_control.game_count,
                }
                self.last_snapshot_time = current_time
                self.handle_screenshot(state, reward, info)
            time.sleep(0.01)

    def handle_screenshot(self, state, reward, info):
        pass

    def _capture_state(self):
        frame, _ = self.game_control.capture_screen()
        return self._preprocess_state(frame)

    def reset(self):
        initial_state = self.game_control.reset_game()
        self.reward_system.reset()
        self.last_snapshot_time = time.time()
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
        self.last_snapshot_time = time.time()
        return self._preprocess_state(processed_frame), reward, done, info

    def _preprocess_state(self, state):
        if state.shape != (480, 640, 3):
            state = np.transpose(state, (1, 0, 2))
        return state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.running = False
        self.screenshot_thread.join()
