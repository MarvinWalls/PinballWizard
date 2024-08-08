import cv2
import numpy as np
import time
import logging
import win32gui
from mss import mss
from frame_processor import process_frame
from keyboard_actions import perform_action, press_enter, press_f2
from reward_system import RewardSystem
import tensorflow as tf
from datetime import datetime
import os

class GameControl:
    def __init__(self, window_title, templates_directory, screenshot_dir):
        self.window_title = window_title
        self.templates_directory = templates_directory
        self.screenshot_dir = screenshot_dir
        self.DATA_FILE_PATH = r"C:\Users\marvi\Pinball Wizard\GameplayData\data.json"
        self.reward_system = RewardSystem()
        self.action_interval = 0.025  # Minimum time interval between actions
        self.last_action = {'action': 'no_action', 'time': time.time()}  # Tracks the last action and time

        self.previous_ball_count = 1  # Assuming game starts with 1 ball
        self.cumulative_reward = 0
        self.game_count = 1
        logging.info('GameControl initialized')

    def perform_action(self, action):
        perform_action(action)

        current_frame, timestamp = self.capture_screen()
        if current_frame is None:
            current_frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Use a blank frame if capture fails.

        processed_frame, self.last_action = process_frame(
            current_frame, self.window_title, self.reward_system, self.last_action,
            self.action_interval, self.screenshot_dir, self.DATA_FILE_PATH, timestamp
        )

        # Ensure the processed frame has the correct shape
        processed_frame = self.ensure_correct_shape(processed_frame)

        # Extract current score and ball count
        current_score = self.get_current_score(processed_frame)
        current_ball_count = self.get_current_ball_count(processed_frame)

        # Calculate the reward using the reward system
        reward = self.reward_system.calculate_reward(current_score, current_ball_count)

        # Check if game is reset
        if current_ball_count == 0 and self.previous_ball_count > 0:
            self.cumulative_reward = 0
            self.game_count += 1
            self.reset_game()

        self.previous_ball_count = current_ball_count

        # Update cumulative reward
        self.cumulative_reward += reward

        done = self.evaluate_game_state(processed_frame)

        return processed_frame, reward, done, {
            'processed_frame': processed_frame,
            'ball_count': current_ball_count,
            'score': current_score,
            'game_count': self.game_count,
            'cumulative_reward': self.cumulative_reward
        }

    def reset_game(self):
        press_f2()
        time.sleep(1)
        press_enter()
        time.sleep(1)

        initial_state, _ = self.capture_screen()
        return initial_state

    def evaluate_game_state(self, processed_frame):
        # This can be updated based on your specific game over conditions.
        return False

    def get_current_score(self, frame):
        # Placeholder for actual score extraction logic
        return self.reward_system.previous_score

    def get_current_ball_count(self, frame):
        # Placeholder for actual ball count extraction logic
        return self.reward_system.previous_ball_count

    def capture_screen(self):
        window_handle = win32gui.FindWindow(None, self.window_title)
        if not window_handle:
            print(f"Window with title '{self.window_title}' not found.")
            return None, None

        x1, y1, x2, y2 = win32gui.GetWindowRect(window_handle)

        with mss() as sct:
            monitor = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
            screen = sct.grab(monitor)
            screen_np = np.array(screen)[:, :, :3]  # Discard the alpha channel if present
            screen_tensor = tf.convert_to_tensor(screen_np, dtype=tf.float32)
            resized_screen_tensor = tf.image.resize(screen_tensor, [240, 320])
            resized_screen_np = resized_screen_tensor.numpy().astype(np.uint8)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")

            return resized_screen_np, timestamp

    def ensure_correct_shape(self, frame):
        # Ensure the frame has the correct shape for the model
        if frame.shape != (240, 320, 3):
            frame = np.stack([frame] * 3, axis=-1) if len(frame.shape) == 2 else frame
            frame = cv2.resize(frame, (320, 240))
        return frame
