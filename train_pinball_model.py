from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from pinball_env import PinballEnv

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        print(f"Reward in TensorboardCallback: {reward}")  # Add this line to print the reward
        self.logger.record('reward', reward)
        return True

def train_model(window_title, templates_directory, screenshot_dir):
    env = DummyVecEnv([lambda: PinballEnv(window_title, templates_directory, screenshot_dir)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("pinball_model")

    # Create a TensorBoard callback
    tensorboard_callback = TensorboardCallback()

    # Initialize model
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./logs/")

    try:
        # Train model with the TensorBoard callback
        model.learn(total_timesteps=10000, callback=tensorboard_callback)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save model and perform any necessary cleanup
        model.save("pinball_model")
        env.close()

    # Load model (optional, for demonstration)
    del model
    model = PPO.load("pinball_model")

    # Evaluate model
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()