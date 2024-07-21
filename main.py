import logging
from train_pinball_model import train_model

if __name__ == "__main__":
    logging.basicConfig(filename='gameplay.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    window_title = "3D Pinball for Windows - Space Cadet"
    templates_directory = r'C:\Users\marvi\Pinball Wizard\templates'
    screenshot_dir = r'C:\Users\marvi\Pinball Wizard\Screenshots'

    # Start training, now also passing the screenshot_dir
    train_model(window_title, templates_directory, screenshot_dir)
