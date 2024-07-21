import logging

class RewardSystem:
    def __init__(self):
        self.previous_score = 0
        self.previous_ball_count = 1
        self.reset()

    def calculate_reward(self, current_score, current_ball_count):
        reward = 0

        # Check for ball loss
        if current_ball_count > self.previous_ball_count:
            reward -= 1000  # Penalty for losing a ball
            print(f"Ball lost. Current ball count: {current_ball_count}, Previous ball count: {self.previous_ball_count}")
            logging.info(f"Ball lost. Current ball count: {current_ball_count}, Previous ball count: {self.previous_ball_count}")

        # Check for score increase
        if current_score > self.previous_score:
            reward += current_score - self.previous_score
            print(f"Score increased. Current score: {current_score}, Previous score: {self.previous_score}")
            logging.info(f"Score increased. Current score: {current_score}, Previous score: {self.previous_score}")

        # Check for score reset
        if current_score < self.previous_score:
            print(f"Score reset. Current score: {current_score}, Previous score: {self.previous_score}")
            logging.info(f"Score reset. Current score: {current_score}, Previous score: {self.previous_score}")

        print(f"Calculated reward: {reward}")
        logging.info(f"Calculated reward: {reward}")

        # Update the previous score and ball count for the next frame/calculation.
        self.previous_score = current_score
        self.previous_ball_count = current_ball_count

        return reward

    def reset(self):
        self.previous_score = 0
        self.previous_ball_count = 1
        print("RewardSystem reset.")
        logging.info("RewardSystem reset.")
