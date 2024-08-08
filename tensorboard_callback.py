import os
import logging
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import cv2  # Import OpenCV
import tensorflow as tf  # Import TensorFlow

class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir='./tensorboard_logs/', verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.step = 0
        self.cumulative_reward = 0
        self.start_time = time.time()
        self.previous_score = 0
        self.previous_ball_count = 1
        self.previous_game_count = 1

    def _on_training_start(self):
        run_id = time.strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(self.log_dir, run_id))
        logging.info(f"TensorBoard log directory: {os.path.join(self.log_dir, run_id)}")

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]
        action = self.locals['actions'][0]
        done = self.locals['dones'][0]
        info = self.locals['infos'][0]

        # Extract the current score, ball count, and game count from info
        current_score = info.get('score', 0)
        current_ball_count = info.get('ball_count', 1)
        game_count = info.get('game_count', 1)

        # Calculate reward based on changes in score and ball count
        reward = 0
        if current_score > self.previous_score:
            reward += current_score - self.previous_score  # Reward for increasing score

        if current_ball_count > self.previous_ball_count:
            reward -= 100000  # Penalty for losing a ball

        if game_count > self.previous_game_count:
            reward -= 1000000  # Penalty for increasing game count

        # Update cumulative reward
        self.cumulative_reward += reward

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time

        # Log individual reward, cumulative reward, and elapsed time
        self.writer.add_scalar('Reward', reward, self.step)
        self.writer.add_scalar('Cumulative Reward', self.cumulative_reward, self.step)
        self.writer.add_scalar('Elapsed Time', elapsed_time, self.step)

        # Log action
        self.writer.add_scalar('Action', action, self.step)

        # Log the image (ensure we log every step)
        image = info.get('screenshot')
        if image is not None:
            logging.info(f"Logging image at step {self.step} with shape {image.shape} and dtype {image.dtype}")

            # Ensure it's a NumPy array
            if isinstance(image, tf.Tensor):
                image_np = image.numpy()  # Convert TensorFlow tensor to NumPy array
            else:
                image_np = np.array(image)  # Ensure it's a NumPy array

            # Ensure the image is in the correct format (HWC) and type (uint8)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = image_np.astype(np.uint8)  # Ensure the image is of type uint8
                # Normalize the image to be in the range [0, 1]
                image_np = image_np / 255.0
                self.writer.add_image('Game Screen', image_np, self.step, dataformats='HWC')

        # Log the ball count, score, and game count
        self.writer.add_scalar('Ball Count', current_ball_count, self.step)
        self.writer.add_scalar('Score', current_score, self.step)
        self.writer.add_scalar('Game Count', game_count, self.step)

        # Update previous score, ball count, and game count for the next step
        self.previous_score = current_score
        self.previous_ball_count = current_ball_count
        self.previous_game_count = game_count

        self.step += 1
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
