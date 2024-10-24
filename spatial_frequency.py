import cv2
import numpy as np
import os
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

def fuse_images(imageA, imageB, block_size=8):
    fused_image = np.zeros_like(imageA)
    M, N = imageA.shape

    # Iterate over blocks of the image
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            # Extract regions (blocks)
            regionA = imageA[i:i+block_size, j:j+block_size]
            regionB = imageB[i:i+block_size, j:j+block_size]

            # Compute spatial frequencies for the regions
            SF_A = spatial_frequency(regionA)
            SF_B = spatial_frequency(regionB)

            # Assign the pixel block from the image with the higher spatial frequency
            if SF_A < SF_B:
                fused_image[i:i+block_size, j:j+block_size] = regionA
            else:
                fused_image[i:i+block_size, j:j+block_size] = regionB

    return fused_image

for i in range(956):
    image1 = cv2.imread(os.path.join('polar_tt_predict_C_raw', str(i) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(os.path.join('polar_vp_predict_C_raw', str(i) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image3 = fuse_images(image1,image2)
    result = cv2.imread(os.path.join('label', str(i) + '.tif'), cv2.IMREAD_GRAYSCALE)

    cv2.imwrite(os.path.join('compare', str(i) + '_tt.png'), image1)
    cv2.imwrite(os.path.join('compare', str(i) + '_vp.png'), image2)
    cv2.imwrite(os.path.join('compare', str(i) + '_sf.png'), image3)
    cv2.imwrite(os.path.join('compare', str(i) + '_label.png'), result)

    