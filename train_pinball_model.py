import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from game_control import GameControl

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        print(f"Reward in TensorboardCallback: {reward}")
        self.logger.record('reward', reward)
        return True

def create_env(window_title, templates_directory, screenshot_dir):
    from pinball_env import PinballEnv  # Import here to avoid circular dependency
    return DummyVecEnv([lambda: PinballEnv(window_title, templates_directory, screenshot_dir)])

def train_policy(env, policy, total_timesteps=10000, tensorboard_log=None):
    model = PPO(policy, env, verbose=1, tensorboard_log=tensorboard_log)
    try:
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        model.save("interrupted_model")
    return model

def save_model(model, filename="pinball_model"):
    model.save(filename)
    logging.info(f"Model saved as {filename}")

def load_model(filename="pinball_model"):
    return PPO.load(filename)

def evaluate_model(model, env, game_control):
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()
        game_control.perform_action(action)

def train_model(window_title, templates_directory, screenshot_dir):
    env = create_env(window_title, templates_directory, screenshot_dir)
    game_control = GameControl(window_title, templates_directory, screenshot_dir)

    model = train_policy(env, "MlpPolicy", total_timesteps=10000)
    save_model(model)

    tensorboard_callback = TensorboardCallback()
    model = train_policy(env, "CnnPolicy", total_timesteps=10000, tensorboard_log="./logs/")
    try:
        model.learn(total_timesteps=10000, callback=tensorboard_callback)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        save_model(model, filename="interrupted_model")
    finally:
        save_model(model)
        env.close()

    model = load_model()
    evaluate_model(model, env, game_control)

if __name__ == "__main__":
    logging.basicConfig(filename='gameplay.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    window_title = "3D Pinball for Windows - Space Cadet"
    templates_directory = r'C:\Users\marvi\Pinball Wizard\templates'
    screenshot_dir = r'C:\Users\marvi\Pinball Wizard\Screenshots'

    # Start training, now also passing the screenshot_dir
    train_model(window_title, templates_directory, screenshot_dir)
