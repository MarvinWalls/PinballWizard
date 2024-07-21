from train_pinball_model import train_model
from logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()

    window_title = "3D Pinball for Windows - Space Cadet"
    templates_directory = r'C:\Users\marvi\Pinball Wizard\templates'
    screenshot_dir = r'C:\Users\marvi\Pinball Wizard\Screenshots'  # Define the directory to save screenshots

    # Start training, now also passing the screenshot_dir
    train_model(window_title, templates_directory, screenshot_dir)
