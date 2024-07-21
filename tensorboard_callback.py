import logging
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir='./tensorboard_logs/', verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        self.previous_score = 0
        self.previous_ball_count = 3
        self.cumulative_reward = 0

    def _on_step(self) -> bool:
        action = self.locals['actions'][0]
        info = self.locals['infos'][0]

        # Get current score and ball count
        current_score = info.get('score', self.previous_score)
        current_ball_count = info.get('ball_count', self.previous_ball_count)

        # Calculate step reward based on score and ball count changes
        step_reward = (current_score - self.previous_score) - (self.previous_ball_count - current_ball_count) * 1000

        # Update cumulative reward
        self.cumulative_reward += step_reward

        # Update previous score and ball count
        self.previous_score = current_score
        self.previous_ball_count = current_ball_count

        # Log step reward
        self.writer.add_scalar('Step Reward', step_reward, self.step)
        logging.info(f"Logged step reward: {step_reward} at step: {self.step}")

        # Log cumulative reward
        self.writer.add_scalar('Cumulative Reward', self.cumulative_reward, self.step)
        logging.info(f"Logged cumulative reward: {self.cumulative_reward} at step: {self.step}")

        # Log action
        self.writer.add_scalar('Action', action, self.step)
        logging.info(f"Logged action: {action} at step: {self.step}")

        # Log the image
        image = info.get('screenshot')
        if image is not None:
            self.writer.add_image('Game Screen', image, self.step, dataformats='HWC')
            logging.info(f"Logged image at step: {self.step}")

        # Log the ball count and score
        if current_ball_count is not None:
            self.writer.add_scalar('Ball Count', current_ball_count, self.step)
            logging.info(f"Logged ball count: {current_ball_count} at step: {self.step}")
        if current_score is not None:
            self.writer.add_scalar('Score', current_score, self.step)
            logging.info(f"Logged score: {current_score} at step: {self.step}")

        self.step += 1
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
