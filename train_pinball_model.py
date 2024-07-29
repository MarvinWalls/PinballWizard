import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorboard_callback import TensorboardCallback
from game_control import GameControl

def create_env(window_title, templates_directory, screenshot_dir):
    from pinball_env import PinballEnv
    return DummyVecEnv([lambda: PinballEnv(window_title, templates_directory, screenshot_dir)])

def train_policy(env, policy, total_timesteps=100000, tensorboard_log=None):
    model = PPO(policy, env, verbose=1, tensorboard_log=tensorboard_log, n_steps=2048)
    callback = TensorboardCallback(tensorboard_log)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
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

def continue_training(env, model, total_timesteps=100000, tensorboard_log=None):
    callback = TensorboardCallback(tensorboard_log)
    try:
        model.set_env(env)
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        model.save("interrupted_model")
    return model

def train_model(window_title, templates_directory, screenshot_dir, continue_from=None):
    env = create_env(window_title, templates_directory, screenshot_dir)
    game_control = GameControl(window_title, templates_directory, screenshot_dir)

    # Ensure each training run has a unique tensorboard log directory
    run_id = len(os.listdir('./tensorboard_logs/')) + 1
    tensorboard_log = f"./tensorboard_logs/PPO_{run_id}"

    if continue_from:
        logging.info(f"Continuing training from {continue_from}")
        model = load_model(continue_from)
        model = continue_training(env, model, total_timesteps=100000, tensorboard_log=tensorboard_log)
    else:
        logging.info(f"Starting training run {run_id} with MlpPolicy")
        model = train_policy(env, "MlpPolicy", total_timesteps=100000, tensorboard_log=tensorboard_log)

    save_model(model, filename=f"pinball_model_run_{run_id}.zip")

    logging.info(f"Training and evaluation completed for run {run_id}")

if __name__ == "__main__":
    logging.basicConfig(filename='gameplay.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    window_title = "3D Pinball for Windows - Space Cadet"
    templates_directory = r'C:\Users\marvi\Pinball Wizard\templates'
    screenshot_dir = r'C:\Users\marvi\Pinball Wizard\Screenshots'

    # Start training, optionally continuing from a saved model
    train_model(window_title, templates_directory, screenshot_dir, continue_from=None)
