import cv2
import os
import datetime

def save_screen(screen, save_directory, file_base_name):
    """
    Save the screen to a file within the save_directory with a timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"{file_base_name}-{timestamp}.png"
    file_path = os.path.join(save_directory, file_name)
    cv2.imwrite(file_path, screen)
    print(f"Saved screen to {file_path}")

# Add more utility functions as needed, such as logging or error handling
