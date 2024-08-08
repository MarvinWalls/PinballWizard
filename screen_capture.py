import tensorflow as tf
import numpy as np
import os
import logging
from datetime import datetime
import win32gui
from mss import mss

# Ensure compatibility with TensorFlow's functions
tf.compat.v1.enable_eager_execution()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Function to capture the screen with a timestamp and resize it to 640x480
def capture_screen(window_title, save_dir):
    window_handle = win32gui.FindWindow(None, window_title)
    if not window_handle:
        logging.error(f"Window with title '{window_title}' not found.")
        return None, None

    x1, y1, x2, y2 = win32gui.GetWindowRect(window_handle)

    with mss() as sct:
        monitor = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
        screen = sct.grab(monitor)
        # Convert screen data to a NumPy array, discard the alpha channel if present
        screen_np = np.array(screen)[:, :, :3]
        # Convert the NumPy array to a TensorFlow tensor, then resize it
        screen_tensor = tf.convert_to_tensor(screen_np, dtype=tf.float32)
        resized_screen_tensor = tf.image.resize(screen_tensor, [480, 640])
        # Convert back to a NumPy array for saving to file
        resized_screen_np = resized_screen_tensor.numpy().astype(np.uint8)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        filename = f"pinball_screenshot_{timestamp}.png"
        save_path = os.path.join(save_dir, filename)

        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Encode the image as PNG and save
        encoded_png = tf.io.encode_png(resized_screen_np)
        tf.io.write_file(save_path, encoded_png)

        logging.info(f"Screenshot saved to {save_path}")

        return resized_screen_np, timestamp

# Usage example
if __name__ == "__main__":
    window_title = "3D Pinball for Windows - Space Cadet"
    save_dir = r"C:/Users/marvi/Pinball Wizard/Screenshots"

    screen, timestamp = capture_screen(window_title, save_dir)
    if screen is not None:
        print("Screen captured, resized, and saved successfully.")
        print(f"Screenshot saved to {save_dir}")
    else:
        print("Failed to capture the screen.")
