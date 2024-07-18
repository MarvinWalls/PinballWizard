import cv2
import os

# Function to perform template matching and return coordinates
def match_template(image, template, method=cv2.TM_CCORR_NORMED):
    result = cv2.matchTemplate(image, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val > 0.7:  # Define a threshold for matching
        return max_loc
    return None

# Function to load templates from the 'templates' directory
def load_templates(templates_dir='templates'):
    templates = {}
    for filename in os.listdir(templates_dir):
        name, _ = os.path.splitext(filename)
        template_path = os.path.join(templates_dir, filename)
        # Load and convert to grayscale
        template_image = cv2.imread(template_path)
        templates[name] = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    return templates

# Function to detect objects in the image using templates
def detect_objects(image, templates):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2 and image.shape[2] == 3:  # Color image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections = {}
    for obj_name, tmpl in templates.items():
        coord = match_template(image, tmpl)
        if coord:
            detections[obj_name] = coord
    return detections

# Function to draw rectangles around detected objects
def draw_detections(image, detections, templates):
    # Convert to color image for display if necessary
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for obj_name, top_left in detections.items():
        if top_left is not None:
            h, w = templates[obj_name].shape
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Use a different color for visibility
    return image

# Example usage
if __name__ == "__main__":
    # Specify the path to your templates directory
    templates_dir = 'templates'
    templates = load_templates(templates_dir)

    # Load a sample image (for testing)
    game_screen_path = 'path_to_your_game_screenshot.png'  # Adjust the path
    game_screen = cv2.imread(game_screen_path)

    # Detect objects
    detections = detect_objects(game_screen, templates)

    # Draw detections on the image
    game_screen_with_detections = draw_detections(game_screen, detections, templates)

    # Show the result
    cv2.imshow('Detected Objects', game_screen_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_high_score(frame, high_score_template, threshold=0.9):
    # Check if frame is already grayscale; otherwise convert
    if len(frame.shape) > 2 and frame.shape[2] == 3:  # Color image
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame  # Assuming frame is already grayscale

    # Perform template matching
    res = cv2.matchTemplate(gray_frame, high_score_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= threshold