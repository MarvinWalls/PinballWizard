import logging
from train_pinball_model import train_model

if __name__ == "__main__":
    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler("gameplay.log"),
            logging.StreamHandler()
        ]
    )

    window_title = "3D Pinball for Windows - Space Cadet"
    templates_directory = r'C:\Users\marvi\Pinball Wizard\templates'
    screenshot_dir = r'C:\Users\marvi\Pinball Wizard\Screenshots'

    # Start training, now also passing the screenshot_dir
    train_model(window_title, templates_directory, screenshot_dir)
