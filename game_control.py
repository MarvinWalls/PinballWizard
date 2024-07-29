import cv2
import numpy as np
import time
import logging
import win32gui
from mss import mss
from frame_processor import process_frame
from keyboard_actions import perform_action, press_enter, press_f2
from reward_system import RewardSystem
from object_detection import load_templates, detect_high_score
from screen_capture import capture_screen
import tensorflow as tf
from datetime import datetime

class GameControl:
    def __init__(self, window_title, templates_directory, screenshot_dir):
        self.window_title = window_title
        self.templates_directory = templates_directory
        self.screenshot_dir = screenshot_dir
        self.DATA_FILE_PATH = r"C:\Users\marvi\Pinball Wizard\GameplayData\data.json"
        self.templates = load_templates(self.templates_directory)
        self.reward_system = RewardSystem()
        self.high_score_template = cv2.imread(f"{self.templates_directory}/high_score_template.png",
                                              cv2.IMREAD_GRAYSCALE)
        if self.high_score_template is None:
            raise FileNotFoundError("The high score template was not found.")

        self.action_interval = 0.05  # Minimum time interval between actions
        self.last_action = {'action': 'no_action', 'time': time.time()}  # Tracks the last action and time

        self.previous_ball_count = 1  # Assuming game starts with 1 ball
        self.cumulative_reward = 0
        self.game_count = 1
        logging.info('GameControl initialized')

    def perform_action(self, action):
        logging.info(f"Performing action: {action}")
        perform_action(action)

        current_frame, timestamp = self.capture_screen()
        if current_frame is None:
            logging.warning("Failed to capture the screen. Using a default blank frame.")
            current_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Use a blank frame if capture fails.

        processed_frame, self.last_action = process_frame(
            self.window_title, self.templates, self.reward_system, self.last_action,
            self.action_interval, self.screenshot_dir, self.DATA_FILE_PATH
        )
        logging.info(f"Processed frame type: {type(processed_frame)}")  # Debug statement

        # Extract current score and ball count
        current_score = self.get_current_score(processed_frame)
        current_ball_count = self.get_current_ball_count(processed_frame)

        # Calculate the reward using the reward system
        reward = self.reward_system.calculate_reward(current_score, current_ball_count)

        # Check if game is reset
        if current_ball_count == 0 and self.previous_ball_count > 0:
            self.cumulative_reward = 0
            self.game_count += 1
            logging.info(f"New game detected, resetting cumulative reward. Game count: {self.game_count}")
            self.reset_game()

        self.previous_ball_count = current_ball_count

        # Update cumulative reward
        self.cumulative_reward += reward

        done = self.evaluate_game_state(processed_frame)
        logging.info(f"Action: {action}, Reward: {reward}, Cumulative Reward: {self.cumulative_reward}, Done: {done}")
        return processed_frame, reward, done, {
            'processed_frame': processed_frame,
            'ball_count': current_ball_count,
            'score': current_score,
            'game_count': self.game_count,
            'cumulative_reward': self.cumulative_reward
        }

    def reset_game(self):
        logging.info("Resetting game...")
        press_f2()
        time.sleep(1)
        press_enter()
        time.sleep(1)

        initial_state, _ = self.capture_screen()
        return initial_state

    def evaluate_game_state(self, processed_frame):
        detected = detect_high_score(processed_frame, self.high_score_template, threshold=0.9)
        if detected:
            logging.info("High score detected. Ending game.")
            return True
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
            resized_screen_tensor = tf.image.resize(screen_tensor, [480, 640])
            resized_screen_np = resized_screen_tensor.numpy().astype(np.uint8)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
            filename = f"pinball_screenshot_{timestamp}.png"
            save_path = tf.io.gfile.join(self.screenshot_dir, filename)

            if not tf.io.gfile.exists(self.screenshot_dir):
                tf.io.gfile.makedirs(self.screenshot_dir)

            encoded_png = tf.io.encode_png(resized_screen_np)
            tf.io.write_file(save_path, encoded_png)

            return resized_screen_np, timestamp
