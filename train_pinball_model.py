import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorboard_callback import TensorboardCallback  # Import from the new file
from game_control import GameControl

def create_env(window_title, templates_directory, screenshot_dir):
    from pinball_env import PinballEnv
    return DummyVecEnv([lambda: PinballEnv(window_title, templates_directory, screenshot_dir)])

def train_policy(env, policy, total_timesteps=10000, tensorboard_log=None):
    model = PPO(policy, env, verbose=1, tensorboard_log=tensorboard_log)
    try:
        model.learn(total_timesteps=total_timesteps, callback=TensorboardCallback(tensorboard_log))
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

    # Ensure each training run has a unique tensorboard log directory
    run_id = len(os.listdir('./tensorboard_logs/')) + 1
    tensorboard_log = f"./tensorboard_logs/PPO_{run_id}"

    logging.info(f"Starting training run {run_id} with MlpPolicy")
    model = train_policy(env, "MlpPolicy", total_timesteps=10000, tensorboard_log=tensorboard_log)
    save_model(model)

    logging.info(f"Starting training run {run_id} with CnnPolicy")
    model = train_policy(env, "CnnPolicy", total_timesteps=10000, tensorboard_log=tensorboard_log)
    save_model(model, filename="pinball_model_final")

    model = load_model(filename="pinball_model_final")
    evaluate_model(model, env, game_control)
    logging.info(f"Training and evaluation completed for run {run_id}")

if __name__ == "__main__":
    logging.basicConfig(filename='gameplay.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    window_title = "3D Pinball for Windows - Space Cadet"
    templates_directory = r'C:\Users\marvi\Pinball Wizard\templates'
    screenshot_dir = r'C:\Users\marvi\Pinball Wizard\Screenshots'

    # Start training, now also passing the screenshot_dir
    train_model(window_title, templates_directory, screenshot_dir)