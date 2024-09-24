import cv2
import numpy as np
import os

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
    
def overlap_percentage(img1, img2):
    overlap = np.sum((img1 > 0) & (img2 > 0))
    img1_non_black = np.sum(img1 > 0)
    img2_non_black = np.sum(img2 > 0)
    
    if img1_non_black == 0 or img2_non_black == 0:
        return 0
    
    overlap_percentage = overlap / min(img1_non_black, img2_non_black)
    return overlap_percentage

def fuse_images(image1, image2):
    if overlap_percentage(image1, image2) > 0.95:
        image1[image1 < 150] = 0
        image1[image1 > 150] = 1
        return image1*255
    # Normalize pixel values to range [0, 1]
    image1_norm = image1.astype(float) / 255.0
    image2_norm = image2.astype(float) / 255.0

    if np.sum(image1_norm) < 20:
        return image2_norm*255
    elif np.sum(image2_norm) < 20:
        return image1_norm*255
    else:
        return (image1_norm + image2_norm) / 2.0 * 255




count = []
avg_score = 0
score_before = 0

for i in range(956):
    image1 = cv2.imread(os.path.join('polar_tt_predict_C_raw', str(i) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(os.path.join('polar_vp_predict_C_raw', str(i) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    result = cv2.imread(os.path.join('label', str(i) + '.tif'), cv2.IMREAD_GRAYSCALE)
    
    img1 = image1
    img2 = image2
    fused_image = fuse_images(img1, img2)
    
    score1 = score(result, image1)
    score2 = score(result, image2)
    score_new = score(result, fused_image)
    avg_score += score_new
    score_before += np.maximum(score1, score2)

    print(score1)
    print(score2)
    print("The new IOU: ", score_new)
        
    if score_new > score1 and score_new > score2:
        count.append(i)
        cv2.imwrite(os.path.join('fused', str(i) + '_tt.png'), image1)
        cv2.imwrite(os.path.join('fused', str(i) + '_vp.png'), image2)
        cv2.imwrite(os.path.join('fused', str(i) + '_predict.png'), fused_image)
        cv2.imwrite(os.path.join('fused', str(i) + '_label.png'), result)

print(count)
print(avg_score / 956)
print(score_before / 956)
import cv2
import numpy as np
import os

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
    
def overlap_percentage(img1, img2):
    overlap = np.sum((img1 > 0) & (img2 > 0))
    img1_non_black = np.sum(img1 > 0)
    img2_non_black = np.sum(img2 > 0)
    
    if img1_non_black == 0 or img2_non_black == 0:
        return 0
    
    overlap_percentage = overlap / min(img1_non_black, img2_non_black)
    return overlap_percentage

def fuse_images(image1, image2):
    if overlap_percentage(image1, image2) > 0.95:
        image1[image1 < 150] = 0
        image1[image1 > 150] = 1
        return image1*255
    # Normalize pixel values to range [0, 1]
    image1_norm = image1.astype(float) / 255.0
    image2_norm = image2.astype(float) / 255.0

    if np.sum(image1_norm) < 20:
        return image2_norm*255
    elif np.sum(image2_norm) < 20:
        return image1_norm*255
    else:
        return (image1_norm + image2_norm) / 2.0 * 255




count = []
avg_score = 0
score_before = 0

for i in range(956):
    image1 = cv2.imread(os.path.join('polar_tt_predict_C_raw', str(i) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(os.path.join('polar_vp_predict_C_raw', str(i) + '_predict.png'), cv2.IMREAD_GRAYSCALE)
    result = cv2.imread(os.path.join('label', str(i) + '.tif'), cv2.IMREAD_GRAYSCALE)
    
    img1 = image1
    img2 = image2
    fused_image = fuse_images(img1, img2)
    
    score1 = score(result, image1)
    score2 = score(result, image2)
    score_new = score(result, fused_image)
    avg_score += score_new
    score_before += np.maximum(score1, score2)

    print(score1)
    print(score2)
    print("The new IOU: ", score_new)
        
    if score_new > score1 and score_new > score2:
        count.append(i)
        cv2.imwrite(os.path.join('1', str(i) + '_tt.png'), image1)
        cv2.imwrite(os.path.join('1', str(i) + '_vp.png'), image2)
        cv2.imwrite(os.path.join('1', str(i) + '_predict.png'), fused_image)
        cv2.imwrite(os.path.join('1', str(i) + '_label.png'), result)

print(count)
print(avg_score / 956)
print(score_before / 956)
