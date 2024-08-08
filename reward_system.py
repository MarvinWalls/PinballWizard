import logging

class RewardSystem:
    def __init__(self):
        self.previous_score = 0
        self.previous_ball_count = 1
        self.reset()

    def calculate_reward(self, current_score, current_ball_count):
        reward = 0

        if current_ball_count > self.previous_ball_count:
            reward -= 100000  # Penalty for losing a ball
        if current_score > self.previous_score:
            reward += current_score - self.previous_score
        if current_score < self.previous_score:
            reward -= current_score - self.previous_score

        print(f"Calculated reward: {reward}")
        logging.info(f"Calculated reward: {reward}")

        self.previous_score = current_score
        self.previous_ball_count = current_ball_count

        return reward

    def reset(self):
        self.previous_score = 0
        self.previous_ball_count = 1
        print("RewardSystem reset.")
        logging.info("RewardSystem reset.")
