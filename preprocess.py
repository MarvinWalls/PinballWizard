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


def preprocess_for_ocr(img, coords):
    # Extract the region of interest (ROI) from the image based on the given coordinates
    x, y, w, h = coords
    cropped_img = img[y:y + h, x:x + w]

    # Check if the cropped image is already in grayscale
    if cropped_img.ndim == 2:
        gray_img = cropped_img
    else:
        # Convert the cropped image to grayscale if it has more than one channel
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale image
    adaptive_thresh_img = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Resize the processed image to enhance OCR accuracy
    resized_img = cv2.resize(adaptive_thresh_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Optionally save the image for debugging
    save_image(resized_img, "Debugging", 'debug_preprocessed_for_ocr.png')

    return resized_img