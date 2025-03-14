import cv2
import os
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import square
from scipy.signal import convolve2d
from skimage.util import img_as_ubyte
# def rf_cf(image):
#     image = image.astype(np.float32)
#     M, N = image.shape

#     RF = np.sqrt(
#         (1 / (M * N)) * np.sum(
#             [(image[m, n] - image[m, n - 1]) ** 2 for m in range(0, M) for n in range(1, N)]
#         )
#     )
#     CF = np.sqrt(
#         (1 / (M * N)) * np.sum(
#             [(image[m, n] - image[m - 1, n]) ** 2 for m in range(1, M) for n in range(0, N)]
#         )
#     )
#     return RF, CF

# def spatial_frequency(image):
#     RF, CF = rf_cf(image)
#     SF = np.sqrt(RF**2 + CF**2)
#     return SF

def normalized_local_entropy(image, window_size):
    """Compute the local entropy of an image."""
    image_uint8 = img_as_ubyte(image)
    return entropy(image_uint8, square(window_size))

def local_contrast(image, window_size):
    """Compute the local contrast of an image."""
    local_mean = convolve2d(image, np.ones((window_size, window_size)), mode='same') / (window_size ** 2)
    patch_diff = np.square(image - local_mean)
    local_contrast = np.sqrt(convolve2d(patch_diff, np.ones((window_size, window_size)), mode='same') / (window_size ** 2))
    return local_contrast

def compute_weights(image1, image2, window_size):
    """Compute weights for fusion based on edge detection, spatial frequency, and local contrast."""
    edges1 = cv2.Sobel(image1, cv2.CV_64F, 1, 1, ksize=3)
    edges2 = cv2.Sobel(image2, cv2.CV_64F, 1, 1, ksize=3)
    edges1 = cv2.normalize(edges1, None, 0, 1, cv2.NORM_MINMAX)
    edges2 = cv2.normalize(edges2, None, 0, 1, cv2.NORM_MINMAX)

    # Spatial frequency
    # sf1 = spatial_frequency(image1)
    # sf2 = spatial_frequency(image2)
    # sf1_weight = sf1 / (sf1 + sf2 + 1e-9)
    # sf2_weight = sf2 / (sf1 + sf2 + 1e-9)

    # Local contrast and entropy
    contrast1 = local_contrast(image1, window_size)
    contrast2 = local_contrast(image2, window_size)
    entropy1 = normalized_local_entropy(image1, window_size)
    entropy2 = normalized_local_entropy(image2, window_size)

    # Combine weights
    weight1 = (contrast1 + entropy1) / 2
    weight2 = (contrast2 + entropy2) / 2

    # Normalize weights
    weight1 = weight1 / (weight1 + weight2 + 1e-9)
    weight2 = weight2 / (weight1 + weight2 + 1e-9)

    return weight1, weight2

def gaussian_pyramid(image, levels):
    """Build a Gaussian pyramid for a grayscale image."""
    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def laplacian_pyramid(image, levels):
    """Build a Laplacian pyramid for a grayscale image."""
    gaussian_pyr = gaussian_pyramid(image, levels)
    laplacian_pyr = []
    for i in range(levels - 1):
        upsampled = cv2.pyrUp(gaussian_pyr[i + 1])
        laplacian = cv2.subtract(gaussian_pyr[i], upsampled)
        laplacian_pyr.append(laplacian)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

def fuse_laplacian_pyramids(lap_pyr1, lap_pyr2, weight1_pyr, weight2_pyr):
    """Fuse two Laplacian pyramids using weighted fusion."""
    fused_pyr = []
    for l1, l2, w1, w2 in zip(lap_pyr1, lap_pyr2, weight1_pyr, weight2_pyr):
        fused = w1 * l1 + w2 * l2
        fused_pyr.append(fused)
    return fused_pyr

def collapse_pyramid(laplacian_pyr):
    """Reconstruct an image from a Laplacian pyramid."""
    image = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        image = cv2.pyrUp(image)
        image = cv2.add(image, laplacian_pyr[i])
    return image

def classical_gaussian_kernel(k, sigma):
    w = np.linspace(-(k - 1) / 2, (k - 1) / 2, k)
    x, y = np.meshgrid(w, w)
    kernel = 0.5*np.exp(-0.5*(x**2 + y**2)/(sigma**2))/(np.pi*sigma**2)
    return kernel

def smooth_gaussian_kernel(a):
    w = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
    kernel = np.outer(w, w)
    return kernel

def fuse(image1, image2, levels=5, window_size=7):
    image1 = image1.astype(np.float32) / 255.0
    image2 = image2.astype(np.float32) / 255.0

    lap_pyr1 = laplacian_pyramid(image1,levels)
    lap_pyr2 = laplacian_pyramid(image2,levels)

    # Compute weights for each level
    weight1_pyr = []
    weight2_pyr = []
    for l1, l2 in zip(lap_pyr1, lap_pyr2):
        w1, w2 = compute_weights(l1, l2, window_size)
        weight1_pyr.append(w1)
        weight2_pyr.append(w2)

    fused_pyr = fuse_laplacian_pyramids(lap_pyr1, lap_pyr2, weight1_pyr, weight2_pyr)

    fused_image = collapse_pyramid(fused_pyr)
    fused_image = cv2.normalize(fused_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return fused_image


l = np.arange(0, 956)
for i in range(len(l)):
    image1 = cv2.imread(os.path.join('polar_tt_predict_C_raw', str(l[i]) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(os.path.join('polar_vp_predict_C_raw', str(l[i]) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image3 = fuse(image1,image2)
    #image4 = fuse(image1,image3)
    cv2.imwrite(os.path.join('LP', str(l[i]) + '.png'), image3)