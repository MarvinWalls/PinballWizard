import os
import logging
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import time

class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir='./tensorboard_logs/', verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        self.cumulative_reward = 0
        self.start_time = time.time()
        self.previous_ball_count = 1  # Initial ball count assumption

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        obs = self.locals['new_obs'][0]
        action = self.locals['actions'][0]
        done = self.locals['dones'][0]
        info = self.locals['infos'][0]

        # Extract the current score and ball count from info
        current_score = info.get('score', 0)
        current_ball_count = info.get('ball_count', 1)
        game_count = info.get('game_count', 1)

        # Calculate reward based on score and ball count
        if current_ball_count < self.previous_ball_count:
            reward = current_score - self.cumulative_reward - 1000000  # Penalize for losing a ball
        else:
            reward = current_score - self.cumulative_reward

        # Update cumulative reward
        self.cumulative_reward += reward

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time

        # Log individual reward, cumulative reward, and elapsed time
        self.writer.add_scalar('Reward', reward, self.step)
        self.writer.add_scalar('Cumulative Reward', self.cumulative_reward, self.step)
        self.writer.add_scalar('Elapsed Time', elapsed_time, self.step)
        logging.info(f"Logged reward: {reward}, cumulative reward: {self.cumulative_reward}, elapsed time: {elapsed_time} at step: {self.step}")

        # Log action
        self.writer.add_scalar('Action', action, self.step)
        logging.info(f"Logged action: {action} at step: {self.step}")

        # Log the image
        image = info.get('screenshot')
        if image is not None:
            self.writer.add_image('Game Screen', image, self.step, dataformats='HWC')
            logging.info(f"Logged image at step: {self.step}")

        # Log the ball count, score, and game count
        if ball_count is not None:
            self.writer.add_scalar('Ball Count', current_ball_count, self.step)
            logging.info(f"Logged ball count: {current_ball_count} at step: {self.step}")
        if score is not None:
            self.writer.add_scalar('Score', current_score, self.step)
            logging.info(f"Logged score: {current_score} at step: {self.step}")
        if game_count is not None:
            self.writer.add_scalar('Game Count', game_count, self.step)
            logging.info(f"Logged game count: {game_count} at step: {self.step}")

        self.step += 1
        self.previous_ball_count = current_ball_count  # Update previous ball count
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
