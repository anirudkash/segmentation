import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

rows = 16
cols = 16

images = []
segmented_images = []
#path = '/home/anirud/Desktop/SemanticSeg/gq-segmented-road'
path = '/home/anirud/Desktop/SemanticSeg/gq-segmented-terrain'
sorted_direc = sorted(os.listdir(path))

for curr_img in sorted_direc:
    print('curr_img: ', curr_img)
    img = os.path.join(path, curr_img)
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    segmented_images.append(img)

#path_patches = '/home/anirud/Desktop/SemanticSeg/gq-patches'
path_patches = '/home/anirud/Desktop/SemanticSeg/alc-resized'
sorted_direc = sorted(os.listdir(path_patches))

for curr_img in sorted_direc:
    #print('curr_img: ', curr_img)
    img = os.path.join(path_patches, curr_img)
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    images.append(img)

segmented_images = np.array(segmented_images)
images = np.array(images)

seg_images_rows = []
images_rows = []

for i in range(0, len(segmented_images), 4): # 4 was prev 16
    patch = segmented_images[i:i+4]
    row = cv2.hconcat(patch)
    seg_images_rows.append(row)

seg_output = cv2.vconcat(seg_images_rows)

for i in range(0, len(images), 4): # 4 was previously 16
    patch = images[i:i+4]
    row = cv2.hconcat(patch)
    images_rows.append(row)

output = cv2.vconcat(images_rows)

plt.figure(figsize=(12,8))

plt.subplot(231)
plt.title('Aerial Image of GQ')
plt.imshow(output)
plt.axis('off')
plt.subplot(232)
plt.title('Segmented Output of GQ (road model)')
plt.imshow(seg_output)
plt.axis('off')
plt.show()