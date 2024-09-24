import cv2
import numpy as np
import os
import math


# https://ieeexplore.ieee.org/abstract/document/10478222
# https://ieeexplore.ieee.org/abstract/document/9171994
# Maybe we could implement some of the algorithm/math


def score(label, predict):
    overlap = 0
    all = 0
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] >= 150 and predict[i][j] >= 150:
                overlap += 1
                all += 1
            if (label[i][j] >= 150) != (predict[i][j] >= 150):
                all += 1
    if all == 0:
        return 0
    else:
        return overlap / all

def fuse_images(image1, image2):
    # Normalize pixel values to range [0, 1]
    image1_norm = image1.astype(float) / 255.0
    image2_norm = image2.astype(float) / 255.0
        
    fused_image = (image1_norm + image2_norm) / 2.0
    if np.sum(image1_norm) < 20:
        fused_image = image2_norm
    if np.sum(image2_norm) < 20:
        fused_image = image1_norm
    fused_image = sharp_edge(fused_image,image1,image2)
    # taking the middle pixel value at each pixel position
    # fused_image = (image1_norm + image2_norm) / 2.0
    count = 0
    gray = 0
    for i in range(fused_image.shape[0]):
        for j in range(fused_image.shape[1]):
            if fused_image[i][j] > 0:
                count += 1
                gray += fused_image[i][j]
    if count > 0:
        threshold = gray/count * 270
    else:
        return fused_image

    # <threshold, ignore
    fused_image[fused_image < threshold / 255.0] = 0
    fused_image[fused_image > threshold / 255.0] = 1
    
    # Convert back to uint8 and scale to 0-255 range
    fused_image = (fused_image * 255).astype(np.uint8)
    
    return fused_image

def create_mask(image):
    h, w = image.shape
    center = (w // 2, h // 2)
    radius = min(center[0], center[1], w - center[0], h - center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    
    mask = dist_from_center <= radius
    return mask



def sharp_edge(image, image1, image2):
    """_identify sharp edges and make the "unsure" part white, "sure" part black_

    Args:
        image (_type_): _description_
        image1 (_type_): _description_
        image2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    mask1 = create_mask(image1)
    masked_image1 = image1 * mask1.astype(np.uint8)
    edge1 = cv2.Canny(masked_image1, 100, 200)

    mask2 = create_mask(image2)
    masked_image2 = image2 * mask2.astype(np.uint8)
    edge2 = cv2.Canny(masked_image2, 100, 200)

    def modify(image, masked_image, edge):
        output_image = np.zeros_like(image)
        for i in range(1, edge.shape[0] - 1):
            for j in range(1, edge.shape[1] - 1):
                if edge[i, j] > 0:
                    intensity = masked_image[i, j]
                    surrounding_intensity = masked_image[i-1:i+2, j-1:j+2].flatten()
                    surrounding_intensity = surrounding_intensity[surrounding_intensity != intensity]
                    if surrounding_intensity.size == 0:
                        continue
                    diff = np.abs(intensity - surrounding_intensity)
                    if np.max(diff) > (2/3) * intensity:
                        # Mark the sharp pixel and its sure side as white in the output image
                        output_image[i, j] = 255
                        if intensity > surrounding_intensity.mean():
                            output_image[i-1:i+2, j-1:j+2] = 255
                        else:
                            # Set unsure side to black
                            output_image[i-1:i+2, j-1:j+2][output_image[i-1:i+2, j-1:j+2] == 0] = 0
                            output_image[i, j] = 0

        # Combine the sharp parts with the original image
        image[np.where(output_image == 255)] = 255
        image[np.where(output_image == 0)] = 0
        return image

    image = modify(image, masked_image1, edge1)
    image = modify(image, masked_image2, edge2)

    return image

