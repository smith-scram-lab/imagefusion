import cv2
import numpy as np
import os
from scipy.ndimage import convolve

#https://www.sciencedirect.com/science/article/pii/S1350449519303822
### w_i(x, y) = \exp\left(-\frac{(I_i(x, y) - 0.5)^2}{2\sigma^2}\right)

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



def fuse_image(image1, image2, r=0.8, S=7, epsilon=1e-8):
    def local_average(image):
        kernel = np.ones((S, S)) / (S**2)
        return convolve(image, kernel, mode='reflect')

    def calculate_weights(image, middle_gray=0.5, sigma=0.3):
        avg_gray = local_average(image)
        return np.exp(-((avg_gray - middle_gray)**2) / (2 * sigma**2))

    image1_norm = image1.astype(np.float32) / 255.0
    image2_norm = image2.astype(np.float32) / 255.0

    weight1 = calculate_weights(image1_norm)
    weight2 = calculate_weights(image2_norm)

    grayscale_obj = (weight1 * image1_norm + weight2 * image2_norm) / (weight1 + weight2 + epsilon)

    strong_radiation_threshold = r * 0.8
    strong_radiation_regions = (image1_norm > strong_radiation_threshold) | (image2_norm > strong_radiation_threshold)

    gradient1 = np.gradient(image1_norm)
    gradient2 = np.gradient(image2_norm)

    gradient_mag1 = np.sqrt(gradient1[0]**2 + gradient1[1]**2)
    gradient_mag2 = np.sqrt(gradient2[0]**2 + gradient2[1]**2)

    gradient_obj = np.where(strong_radiation_regions, gradient_mag1, gradient_mag2)

    fused_image = grayscale_obj + 0.4 * gradient_obj 

    fused_image = np.clip(fused_image, 0, 1)
    return (fused_image * 255).astype(np.uint8)

def to_binary(image):
    count = 0
    gray = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 0:
                count += 1
                gray += image[i][j]
    if count > 0:
        threshold = gray/count * 270
    else:
        return image

    image[image < threshold / 255.0] = 0
    image[image > threshold / 255.0] = 1
    image = (image * 255).astype(np.uint8)
    
    return image

#for testing 

l = np.arange(0, 956)
for i in range(len(l)):
    image1 = cv2.imread(os.path.join('polar_tt_predict_C_raw', str(l[i]) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(os.path.join('polar_vp_predict_C_raw', str(l[i]) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image3 = fuse_image(image1,image2)
    cv2.imwrite(os.path.join('gradient', str(l[i]) + '.png'), image3)
