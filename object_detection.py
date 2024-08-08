import cv2
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def match_template(image, template, method=cv2.TM_CCORR_NORMED, threshold=0.7):
    """
    Perform template matching and return coordinates if match exceeds threshold.

    :param image: Grayscale image where the template is searched.
    :param template: Grayscale template image.
    :param method: Template matching method.
    :param threshold: Matching threshold.
    :return: Top-left coordinates of the best match if above threshold, otherwise None.
    """
    result = cv2.matchTemplate(image, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val > threshold:
        logging.info(f"Template matched with value {max_val} at location {max_loc}")
        return max_loc
    logging.info(f"Template match value {max_val} below threshold {threshold}")
    return None

def load_templates(templates_dir='templates'):
    """
    Load all templates from the specified directory.

    :param templates_dir: Directory containing template images.
    :return: Dictionary of template names and images.
    """
    templates = {}
    for filename in os.listdir(templates_dir):
        name, _ = os.path.splitext(filename)
        template_path = os.path.join(templates_dir, filename)
        template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template_image is not None:
            templates[name] = template_image
            logging.info(f"Loaded template {name} from {template_path}")
        else:
            logging.warning(f"Failed to load template {name} from {template_path}")
    return templates

def detect_objects(image, templates):
    """
    Detect objects in the image using the provided templates.

    :param image: Grayscale image where objects are searched.
    :param templates: Dictionary of template names and images.
    :return: Dictionary of detected object names and their coordinates.
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections = {}
    for obj_name, tmpl in templates.items():
        coord = match_template(image, tmpl)
        if coord:
            detections[obj_name] = coord
            logging.info(f"Detected {obj_name} at {coord}")
    return detections

def draw_detections(image, detections, templates):
    """
    Draw rectangles around detected objects on the image.

    :param image: Image where detections are drawn.
    :param detections: Dictionary of detected object names and their coordinates.
    :param templates: Dictionary of template names and images.
    :return: Image with detections drawn.
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for obj_name, top_left in detections.items():
        if top_left is not None:
            h, w = templates[obj_name].shape
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
            logging.info(f"Drew rectangle for {obj_name} at {top_left} to {bottom_right}")
    return image

def detect_high_score(frame, high_score_template, threshold=0.9):
    """
    Detect if the high score template is present in the frame.

    :param frame: Image frame where high score is searched.
    :param high_score_template: Template image for high score.
    :param threshold: Matching threshold.
    :return: True if high score is detected, False otherwise.
    """
    if len(frame.shape) > 2 and frame.shape[2] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    res = cv2.matchTemplate(gray_frame, high_score_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    logging.info(f"High score template match value: {max_val}")
    return max_val >= threshold

# Example usage
if __name__ == "__main__":
    templates_dir = 'templates'
    templates = load_templates(templates_dir)

    game_screen_path = 'path_to_your_game_screenshot.png'
    game_screen = cv2.imread(game_screen_path)

    detections = detect_objects(game_screen, templates)
    game_screen_with_detections = draw_detections(game_screen, detections, templates)

    cv2.imshow('Detected Objects', game_screen_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
