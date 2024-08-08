import cv2
import numpy as np
import time
import logging
import random
from keyboard_actions import perform_action
from reward_system import RewardSystem
from data_collection import save_step_data
from preprocess import preprocess_screen, save_preprocessed_screen
from score_detection import read_text_from_area, parse_number_from_text, SCORE_AREA, BALL_COUNT_AREA

SCREENSHOT_DIR = r"C:\Users\marvi\Pinball Wizard\Screenshots"
DATA_FILE_PATH = r"C:\Users\marvi\Pinball Wizard\GameplayData\data.json"

def process_frame(screen, window_title, reward_system, last_action, action_interval, SCREENSHOT_DIR, DATA_FILE_PATH, timestamp):
    if screen is not None:
        try:
            rgb_screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
            preprocessed_screen = preprocess_screen(rgb_screen, width=640, height=480)

            # Ensure the image is in color (3 channels)
            if len(preprocessed_screen.shape) == 2:
                preprocessed_screen = cv2.cvtColor(preprocessed_screen, cv2.COLOR_GRAY2RGB)

            preprocessed_screenshot_path = save_preprocessed_screen(preprocessed_screen, SCREENSHOT_DIR, 'preprocessed_frame', timestamp, quality=75)

            score_text = read_text_from_area(preprocessed_screen, SCORE_AREA)
            ball_count_text = read_text_from_area(preprocessed_screen, BALL_COUNT_AREA)

            logging.info(f"Raw OCR score text: {score_text}, Raw OCR ball count text: {ball_count_text}")

            score = parse_number_from_text(score_text)
            ball_count = parse_number_from_text(ball_count_text)

            reward = reward_system.calculate_reward(score, ball_count)
            logging.info(f"Detected Score: {score}, Detected Ball Count: {ball_count}, Reward: {reward}")

            current_time = time.time()
            if current_time - last_action['time'] > action_interval:
                action = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
                last_action['action'] = action
                last_action['time'] = current_time
            else:
                action = 0  # No action

            perform_action(action)

            save_step_data({
                'timestamp': timestamp,
                'action': action,
                'reward': reward,
                'screenshot_path': preprocessed_screenshot_path
            }, DATA_FILE_PATH)

            return preprocessed_screen, last_action
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8), last_action
    else:
        logging.warning("Screen capture failed.")
        return np.zeros((480, 640, 3), dtype=np.uint8), last_action
