import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math


path = '/home/anirud/Desktop/SemanticSeg/gq-segmented-terrain/'
classes_path = '/home/anirud/Desktop/SemanticSeg/gq-segmented-terrain-classes'
og_img = cv2.imread('/home/anirud/Desktop/SemanticSeg/alc-raw/gq-test.tiff')
#og_img = cv2.resize(og_img, (128, 128))

patch_size = 128
#overlap = 64
overlap = 0
num_rows = 4 # prev 7
num_cols = 4 # prev 7
#total_patches = 49
total_patches = 16

stitched_height = (patch_size-overlap) * (num_rows-1) + patch_size
stitched_width = (patch_size-overlap) * (num_cols-1) + patch_size

stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
classes_image = np.zeros((stitched_height, stitched_width))

for idx in range(total_patches):
    row_idx = idx // num_rows
    col_idx = idx % num_cols

    if idx < 9:
        patch_path = os.path.join(path, f"gq-00{idx+1}.tiff")
        classes_patch_path = os.path.join(classes_path, f"gq-00{idx+1}.tiff")
    elif idx < 99:
        patch_path = os.path.join(path, f"gq-0{idx+1}.tiff")
        classes_patch_path = os.path.join(classes_path, f"gq-0{idx+1}.tiff")
    else:
        patch_path = os.path.join(path, f"gq-{idx+1}.tiff")
        classes_patch_path = os.path.join(classes_path, f"gq-{idx+1}.tiff")
        #raise Exception("implement this;")
    
    segmented_patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
    segmented_raw_patch = cv2.imread(classes_patch_path, cv2.IMREAD_UNCHANGED)

    start_row = row_idx * (patch_size - overlap)
    start_col = col_idx * (patch_size - overlap)
    end_row = start_row + patch_size
    end_col = start_col + patch_size
    #print('stitched_image[end_col]: ',stitched_image[end_col])
    stitched_image[start_row:end_row, start_col:end_col] = segmented_patch
    classes_image[start_row:end_row, start_col:end_col] = segmented_raw_patch


cv2.imwrite('OUTPUT/terrain_model.tiff', stitched_image)
cv2.imwrite('OUTPUT/terrain_model_classes.tiff', classes_image)
plt.figure(figsize=(12,8))
plt.subplot(231)
plt.imshow(og_img)
plt.subplot(232)
plt.imshow(stitched_image)
plt.subplot(233)
plt.imshow(classes_image)
plt.show()
    
# img = np.array([img1, img2])

# row = cv2.hconcat(img)

# plt.imshow(row)
# plt.show()

# plt.figure(figsize=(12,8))

# plt.subplot(231)
# plt.imshow(img1)

# plt.subplot(232)
# plt.imshow(img2)

# plt.subplot(233)
# plt.imshow(row)
# plt.show()
