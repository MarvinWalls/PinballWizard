import cv2
import numpy as np
import time
from frame_processor import process_frame
from keyboard_actions import perform_action, press_enter, press_f2
from reward_system import RewardSystem
from object_detection import load_templates, detect_high_score
from screen_capture import capture_screen


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
        self.action_interval = 0.5
        self.last_action = {'action': 'no_action', 'time': time.time()}

    def perform_action(self, action):
        print(f"Performing action: {action}")
        perform_action(action)

        current_frame, timestamp = capture_screen(self.window_title, self.screenshot_dir)
        if current_frame is None:
            print("Failed to capture the screen. Using a default blank frame.")
            current_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Use a blank frame if capture fails.

        processed_frame, self.last_action = process_frame(
            self.window_title, self.templates, self.reward_system, self.last_action,
            self.action_interval, self.screenshot_dir, self.DATA_FILE_PATH
        )
        print(f"Processed frame type: {type(processed_frame)}")  # Debug statement
        reward, done = self.evaluate_game_state(processed_frame)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        return processed_frame, reward, done

    def reset_game(self):
        print("Resetting game...")
        press_f2()
        time.sleep(1)
        press_enter()
        time.sleep(1)

        initial_state, _ = capture_screen(self.window_title, self.screenshot_dir)
        return initial_state

    def evaluate_game_state(self, processed_frame):
        detected = detect_high_score(processed_frame, self.high_score_template, threshold=0.9)
        if detected:
            print("High score detected. Ending game.")
            return 0, True
        return 0, False
