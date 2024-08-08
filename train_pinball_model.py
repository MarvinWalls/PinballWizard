import os
import logging
from datetime import datetime
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorboard_callback import TensorboardCallback
from game_control import GameControl

def create_env(window_title, templates_directory, screenshot_dir):
    from pinball_env import PinballEnv
    return DummyVecEnv([lambda: PinballEnv(window_title, templates_directory, screenshot_dir)])

def train_policy(env, policy, total_timesteps=256, tensorboard_log=None):
    n_steps = min(2048, total_timesteps)  # Ensure n_steps is not greater than total_timesteps
    model = PPO(policy, env, verbose=1, tensorboard_log=tensorboard_log, n_steps=n_steps)
    callback = TensorboardCallback(log_dir=tensorboard_log)
    logging.info(f"Training policy with total_timesteps={total_timesteps} and n_steps={n_steps}")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        model.save("interrupted_model.zip")
    finally:
        model.save("final_model.zip")
        logging.info("Model saved as final_model.zip")
    return model

def save_model(model, filename="pinball_model.zip"):
    model.save(filename)
    logging.info(f"Model saved as {filename}")

def load_model(filename="pinball_model.zip"):
    try:
        model = PPO.load(filename)
        logging.info(f"Model loaded from {filename}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {filename}: {e}")
        raise

def evaluate_model(model, env, game_control):
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()
        game_control.perform_action(action)

def continue_training(env, model, total_timesteps=256, tensorboard_log=None):
    callback = TensorboardCallback(log_dir=tensorboard_log)
    logging.info(f"Continuing training with total_timesteps={total_timesteps}")
    try:
        model.set_env(env)
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        model.save("interrupted_model.zip")
        logging.info("Model saved as interrupted_model.zip")
    finally:
        model.save("final_model.zip")
        logging.info("Model saved as final_model.zip")
    return model

def train_model(window_title, templates_directory, screenshot_dir, model_filename="final_model.zip", total_timesteps=256):
    env = create_env(window_title, templates_directory, screenshot_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = f"./tensorboard_logs/{run_id}"

    start_time = time.time()

    logging.info(f"Starting training with total_timesteps={total_timesteps}")

    if os.path.exists(model_filename):
        try:
            logging.info(f"Loading model from {model_filename}")
            model = load_model(model_filename)
            model = continue_training(env, model, total_timesteps=total_timesteps, tensorboard_log=tensorboard_log)
        except ValueError as e:
            logging.error(f"Failed to load model from {model_filename}: {e}")
            logging.info("Starting training run with MlpPolicy")
            model = train_policy(env, "MlpPolicy", total_timesteps=total_timesteps, tensorboard_log=tensorboard_log)
    else:
        logging.info("Starting training run with MlpPolicy")
        model = train_policy(env, "MlpPolicy", total_timesteps=total_timesteps, tensorboard_log=tensorboard_log)

    end_time = time.time()

    save_model(model, filename=model_filename)
    logging.info(f"Training and evaluation completed in {end_time - start_time} seconds")
    env.close()  # Ensure the environment is properly closed

    # Ensure the script exits after training
    import sys
    sys.exit()

if __name__ == "__main__":
    logging.basicConfig(filename='gameplay.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    window_title = "3D Pinball for Windows - Space Cadet"
    templates_directory = r'C:\Users\marvi\Pinball Wizard\templates'
    screenshot_dir = r'C:\Users\marvi\Pinball Wizard\Screenshots'

    # Start training, optionally continuing from a saved model
    total_timesteps = 256  # Set the desired number of timesteps for training
    train_model(window_title, templates_directory, screenshot_dir, model_filename="final_model.zip", total_timesteps=total_timesteps)
