import cv2
import random
import time
from object_detection import detect_objects, load_templates, draw_detections
from score_detection import read_text_from_area, parse_number_from_text, SCORE_AREA, BALL_COUNT_AREA
from keyboard_actions import perform_action
from data_collection import save_step_data
from preprocess import preprocess_screen, save_preprocessed_screen
from screen_capture import capture_screen

SCREENSHOT_DIR = r"C:\Users\marvi\Pinball Wizard\Screenshots"
DATA_FILE_PATH = r"C:\Users\marvi\Pinball Wizard\GameplayData\data.json"

def process_frame(window_title, templates, reward_system, last_action, action_interval, SCREENSHOT_DIR, DATA_FILE_PATH):
    screen, timestamp = capture_screen(window_title, SCREENSHOT_DIR)
    if screen is not None:
        rgb_screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
        preprocessed_screen = preprocess_screen(rgb_screen, width=640, height=480)

        preprocessed_screenshot_path = save_preprocessed_screen(preprocessed_screen, SCREENSHOT_DIR, 'preprocessed_frame', timestamp, quality=75)

        detections = detect_objects(preprocessed_screen, templates)
        image_with_detections = draw_detections(preprocessed_screen, detections, templates)

        score_text = read_text_from_area(preprocessed_screen, SCORE_AREA)
        ball_count_text = read_text_from_area(preprocessed_screen, BALL_COUNT_AREA)

        score = parse_number_from_text(score_text)
        ball_count = parse_number_from_text(ball_count_text)

        reward = reward_system.calculate_reward(score, ball_count)
        print(f"Detected Score: {score}, Detected Ball Count: {ball_count}, Reward: {reward}")

        current_time = time.time()
        if current_time - last_action['time'] > action_interval:
            action = random.choice(["press_left_flipper", "release_left_flipper", "press_right_flipper", "release_right_flipper", "press_plunger", "release_plunger"])
            last_action['action'] = action
            last_action['time'] = current_time
        else:
            action = 'no_action'

        perform_action(action)

        save_step_data({
            'timestamp': timestamp,
            'action': action,
            'reward': reward,
            'screenshot_path': preprocessed_screenshot_path
        }, DATA_FILE_PATH)

        cv2.imshow('Detected Objects', image_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False, last_action
    else:
        print("Screen capture failed.")
    return True, last_action


