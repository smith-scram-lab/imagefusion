import cv2
import numpy as np
import math
import os

def is_sharp(image, intensity_threshold=180, edge_threshold=150):
    def create_mask(image):
        # Example mask creation function; adjust as needed
        return np.ones_like(image, dtype=np.uint8)  # Replace with actual mask logic

    mask = create_mask(image)
    masked_image = image * mask.astype(np.uint8)
    
    # Calculate average intensity
    count = np.count_nonzero(masked_image)
    if count == 0:
        return False
    gray = np.sum(masked_image)
    n = gray / count
    
    # Detect edges
    edges = cv2.Canny(masked_image, n, 255)
    
    # Calculate edge statistics
    num_edges = np.sum(edges > 0)
    area = np.sum(masked_image > 0)
    side_length = math.sqrt((4 * area) / math.sqrt(3))
    min_edge = 3 * side_length
    max_edge = 1.9 * min_edge
    
    if num_edges < min_edge or num_edges > max_edge:
        return False

    # Check intensity difference with neighbors
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    edge_response = cv2.filter2D(masked_image, -1, kernel)
    sharpness = np.sum(np.abs(edge_response) > intensity_threshold)
    
    # Return if sharpness is above the threshold
    return sharpness > edge_threshold

def fuse_images(image1, image2):
    def create_sharp_edge_mask(image):
        if is_sharp(image):
            edges = cv2.Canny(image, 100, 200)  # Adjust thresholds if needed
            return edges
        return np.zeros_like(image)

    # Create sharp edge masks
    mask1 = create_sharp_edge_mask(image1)
    mask2 = create_sharp_edge_mask(image2)

    # Combine sharp edges
    sharp_edge = np.where(mask1 > 0, image1, np.where(mask2 > 0, image2, 0))
    
    # Combine the rest of the image content
    rest_of_image = np.where(sharp_edge == 0, (image1 + image2) / 2, sharp_edge)
    
    return rest_of_image



def check(image1, image2, i):
    if is_sharp(image1) or is_sharp(image2):
        cv2.imwrite(os.path.join('group1', f'{i}_tt_predict.png'), image1)
        cv2.imwrite(os.path.join('group1', f'{i}_vp_predict.png'), image2)
filter()

