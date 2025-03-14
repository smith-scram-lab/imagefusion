import cv2
import numpy as np
import os

import numpy as np
import cv2


def rf_cf(image):
    image = image.astype(np.float32)
    M, N = image.shape

    RF = np.sqrt(
        (1 / (M * N)) * np.sum(
            [(image[m, n] - image[m, n - 1]) ** 2 for m in range(0, M) for n in range(1, N)]
        )
    )
    CF = np.sqrt(
        (1 / (M * N)) * np.sum(
            [(image[m, n] - image[m - 1, n]) ** 2 for m in range(1, M) for n in range(0, N)]
        )
    )
    return RF, CF

def spatial_frequency(image):
    RF, CF = rf_cf(image)
    SF = np.sqrt(RF**2 + CF**2)
    return SF

def fuse(image1, image2):
    kernel_size = 7
    edges1 = cv2.Sobel(image1, cv2.CV_64F, 1, 1, ksize=kernel_size)
    edges2 = cv2.Sobel(image2, cv2.CV_64F, 1, 1, ksize=kernel_size)
    edges1 = cv2.normalize(edges1, None, 0, 1, cv2.NORM_MINMAX)
    edges2 = cv2.normalize(edges2, None, 0, 1, cv2.NORM_MINMAX)

    sf1 = spatial_frequency(image1)
    sf2 = spatial_frequency(image2)
    sf1_weight = sf1 / (sf1 + sf2 + 1e-9)  
    sf2_weight = sf2 / (sf1 + sf2 + 1e-9)
    if sf2_weight < sf1_weight/2:
        sf2_weight = sf1_weight/2
    if sf1_weight < sf2_weight/2:
        sf1_weight = sf2_weight/2 

    weight1 = (edges1 / (edges1 + edges2 + 1e-9) + sf1_weight)/2
    weight2 = (edges2 / (edges1 + edges2 + 1e-9) + sf2_weight)/2
    fused_image = weight1 * image1 + weight2 * image2 
    fused_image = cv2.normalize(fused_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return fused_image

# def process_vp(predict):
#     weight = np.sum(predict[predict > 0])
#     weight = weight / (predict.shape[0] * predict.shape[1])
#     if weight == 0:  
#         return np.zeros_like(predict, dtype=np.uint8)
#     scale_factor = 256 / weight
#     predict = cv2.normalize(predict, None, 0, 1, cv2.NORM_MINMAX)
#     new_vp = predict * scale_factor
#     new_vp = np.clip(new_vp, 0, 255).astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     new_vp = clahe.apply(new_vp)
#     return new_vp

l = np.arange(0, 956)
for i in range(len(l)):
    image1 = cv2.imread(os.path.join('polar_tt_predict_C_raw', str(l[i]) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(os.path.join('polar_vp_predict_C_raw', str(l[i]) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image3 = fuse(image1,image2)
    cv2.imwrite(os.path.join('sf', str(l[i]) + '.png'), image3)

#fused_ is skrewed

