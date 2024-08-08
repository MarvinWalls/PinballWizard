import cv2
import numpy as np
import os

def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), image)

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_canny_edge_detection(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def preprocess_screen(screen, width=320, height=240, canny=False):
    if screen is None:
        raise ValueError("Error: Screen is None")

    if not isinstance(screen, np.ndarray):
        raise TypeError("Error converting screen to NumPy array.")

    print(f"Original screen dimensions: {screen.shape}")

    if screen.ndim != 3 or screen.shape[2] != 3:
        raise ValueError(f"Error: Invalid screen shape. Expected (height, width, 3), got {screen.shape}")

    resized_screen = resize_image(screen, width, height)
    print(f"Resized screen dimensions: {resized_screen.shape}")

    grayscale_screen = convert_to_grayscale(resized_screen)
    print(f"Grayscale screen dimensions: {grayscale_screen.shape}")

    if canny:
        edge_screen = apply_canny_edge_detection(grayscale_screen)
        return edge_screen

    return grayscale_screen

def save_preprocessed_screen(image, folder, base_filename, timestamp, quality=95):
    print(f"Saving preprocessed screen with dimensions: {image.shape}")  # Log dimensions
    filepath = os.path.join(folder, f"{base_filename}-{timestamp}.jpg")
    os.makedirs(folder, exist_ok=True)
    if len(image.shape) == 2:  # Image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(filepath, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return filepath

def preprocess_for_ocr(img, coords):
    x, y, w, h = coords
    cropped_img = img[y:y+h, x:x+w]

    if len(cropped_img.shape) == 2 or cropped_img.shape[2] == 1:
        gray_img = cropped_img
    else:
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    alpha = 1.5
    beta = 0
    contrasted_img = cv2.convertScaleAbs(binary_img, alpha=alpha, beta=beta)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(contrasted_img, -1, kernel)

    # Uncomment the line below if you need to save the processed image for debugging
    # save_image(sharpened_img, "Debugging", "debug_preprocessed_for_ocr.png")

    return sharpened_img
