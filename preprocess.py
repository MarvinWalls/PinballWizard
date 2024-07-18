import cv2
import numpy as np
import os

def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), image)

def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def apply_canny_edge_detection(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged_image = cv2.Canny(image, lower, upper)
    return edged_image

def preprocess_screen(screen, width, height, canny=False):
    if screen is None:
        raise ValueError("Error: Screen is None")

    if not isinstance(screen, np.ndarray):
        raise TypeError(f"Error converting screen to NumPy array.")

    if screen.ndim != 3 or screen.shape[2] != 3:
        raise ValueError(f"Error: Invalid screen shape. Expected (height, width, 3), got {screen.shape}")

    resized_screen = resize_image(screen, width, height)
    grayscale_screen = convert_to_grayscale(resized_screen)

    if canny:
        edge_screen = apply_canny_edge_detection(grayscale_screen)
        return edge_screen

    return grayscale_screen

def save_preprocessed_screen(image, folder, base_filename, timestamp, quality=95):
    filepath = os.path.join(folder, f"{base_filename}-{timestamp}.jpg")
    os.makedirs(folder, exist_ok=True)
    # Convert grayscale image back to BGR format for JPEG saving if necessary
    if len(image.shape) == 2:  # Image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(filepath, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return filepath

import cv2
import numpy as np

import cv2
import numpy as np

def preprocess_for_ocr(img, coords):
    # Crop the image to the region of interest
    x, y, w, h = coords
    cropped_img = img[y:y+h, x:x+w]

    # Check if the image has more than one channel
    if len(cropped_img.shape) == 2 or cropped_img.shape[2] == 1:
        gray_img = cropped_img
    else:
        # Convert to grayscale
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply thresholding to get a binary image
    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Increase contrast
    alpha = 1.5  # Simple contrast control
    beta = 0    # Simple brightness control
    contrasted_img = cv2.convertScaleAbs(binary_img, alpha=alpha, beta=beta)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(contrasted_img, -1, kernel)

    # Save the processed image for debugging
    cv2.imwrite("Debugging/debug_preprocessed_for_ocr.png", sharpened_img)

    return sharpened_img


