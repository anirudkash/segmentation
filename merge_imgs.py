import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

path = '/home/anirud/Desktop/SemanticSeg/gq-segmented-terrain/'

patch_size = 128
overlap = 64
num_rows = 7
num_cols = 7
total_patches = 49

stitched_height = (patch_size-overlap) * (num_rows-1) + patch_size
stitched_width = (patch_size-overlap) * (num_cols-1) + patch_size

stitched_image = np.zeros((stitched_height, stitched_width))

for idx in range(total_patches):
    row_idx = idx // num_rows
    col_idx = idx % num_cols

    if idx < 9:
        patch_path = os.path.join(path, f"gq-00{idx+1}.tiff")
        print('patch_path: ', patch_path)
    elif idx < 99:
        patch_path = os.path.join(path, f"gq-0{idx+1}.tiff")
    else:
        raise Exception("implement this")
    
    segmented_patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)

    start_row = row_idx * (patch_size - overlap)
    start_col = col_idx * (patch_size - overlap)
    end_row = start_row + patch_size
    end_col = start_col + patch_size

    stitched_image[start_row:end_row, start_col:end_col] = segmented_patch


plt.imshow(stitched_image)
plt.show()