import cv2
import pytesseract
import re
import os  # Import the os module for file operations
from preprocess import preprocess_for_ocr

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the coordinates for the score and ball count areas
SCORE_AREA = (456, 245, 158, 32)
BALL_COUNT_AREA = (579, 201, 30, 29)

def read_text_from_area(img, coords):
    processed_img = preprocess_for_ocr(img, coords)
    try:
        text = pytesseract.image_to_string(processed_img, config='--psm 7', timeout=10)  # Set a timeout
    except pytesseract.TesseractError as e:
        print(f"Error reading text with Tesseract: {e}")
        text = ""
    print(f"Text read from area {coords}: {text}")
    return text.strip()

def parse_number_from_text(text):
    match = re.search(r'\d+', text)
    if match:
        number = int(match.group())
        print(f"Parsed number from text '{text}': {number}")
        return number
    else:
        print(f"Could not parse number from text '{text}', returning 0")
        return 0  # Ensures a number is always returned, preventing TypeError

# This function is just for debugging purposes, to save the cropped areas as images
def save_debug_images_for_ocr(preprocessed_screen):
    folder = "Debugging"
    save_image(preprocess_for_ocr(preprocessed_screen, SCORE_AREA), folder, "debug_score_area.png")
    save_image(preprocess_for_ocr(preprocessed_screen, BALL_COUNT_AREA), folder, "debug_ball_count_area.png")

# Function to save an image to a specified folder
def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), image)
