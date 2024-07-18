class RewardSystem:
    def __init__(self):
        self.previous_score = 0
        self.previous_ball_count = 1
        # Call reset to initialize your variables
        self.reset()

    def calculate_reward(self, current_score, current_ball_count):
        # Initialize reward
        reward = 0

        # If the ball count has increased, it indicates the loss of a ball.
        if current_ball_count > self.previous_ball_count:
            # Apply a negative reward for losing a ball.
            reward -= 1000  # Or any value that signifies a penalty for losing a ball.
            print(f"Ball lost. Current ball count: {current_ball_count}, Previous ball count: {self.previous_ball_count}")

        # Check for score increase and add to reward.
        if current_score > self.previous_score:
            # Reward the score increase
            reward += current_score - self.previous_score
            print(f"Score increased. Current score: {current_score}, Previous score: {self.previous_score}")

        # If the score reset to 0, it might indicate starting a new game or losing a ball.
        if current_score < self.previous_score:
            # Handle score reset logic here if needed, such as:
            # reward -= (some penalty for game reset)
            print(f"Score reset. Current score: {current_score}, Previous score: {self.previous_score}")
            pass

        print(f"Calculated reward: {reward}")

        # Update the previous score and ball count for the next frame/calculation.
        self.previous_score = current_score
        self.previous_ball_count = current_ball_count

        return reward

    def reset(self):
        # Reset the internal state
        self.previous_score = 0
        self.previous_ball_count = 1
        print("RewardSystem reset.")