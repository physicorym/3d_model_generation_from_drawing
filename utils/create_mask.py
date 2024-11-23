    
import cv2
import numpy as np

def create_mask(image):

    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = cv2.drawContours(np.zeros_like(gray), contours, -1, 255, thickness=cv2.FILLED)

    return mask