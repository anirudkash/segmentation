import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# # # def create_stitched(og_img, path, classes_path, index, path_img=None, dest_direc=None):
# # #     patch_size = 128
# # #     #overlap = 64
# # #     overlap = 0
# # #     num_rows = 4 # prev 7
# # #     num_cols = 4 # prev 7
# # #     #total_patches = 49
# # #     total_patches = 16

# # #     stitched_height = (patch_size-overlap) * (num_rows-1) + patch_size
# # #     stitched_width = (patch_size-overlap) * (num_cols-1) + patch_size

# # #     stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
# # #     classes_image = np.zeros((stitched_height, stitched_width))

# # #     for idx in range(total_patches):
# # #         row_idx = idx // num_rows
# # #         col_idx = idx % num_cols

# # #         patch_path = path[idx]
# # #         classes_patch_path = classes_path[idx]
        
# # #         segmented_patch = cv2.imread(path_img[0]+'/'+patch_path, cv2.IMREAD_UNCHANGED)
# # #         segmented_raw_patch = cv2.imread(path_img[1]+'/'+classes_patch_path, cv2.IMREAD_UNCHANGED)
# # #         #segmented_raw_patch = cv2.imread('gq-segmented-terrain-classes/'+classes_patch_path, cv2.IMREAD_UNCHANGED)

# # #         if index < 10:
# # #             patch_str = str(0)+str(0)+str(index)
# # #         elif index < 100:
# # #             patch_str = str(0)+str(index)
# # #         else:
# # #             patch_str = index

# # #         start_row = row_idx * (patch_size - overlap)
# # #         start_col = col_idx * (patch_size - overlap)
# # #         end_row = start_row + patch_size
# # #         end_col = start_col + patch_size
# # #         #print('stitched_image[end_col]: ',stitched_image[end_col])
# # #         stitched_image[start_row:end_row, start_col:end_col] = segmented_patch
# # #         classes_image[start_row:end_row, start_col:end_col] = segmented_raw_patch

# # #     cv2.imwrite(f'{dest_direc[0]}/gq-{patch_str}.tiff', stitched_image)
# # #     cv2.imwrite(f'{dest_direc[1]}/gq-{patch_str}.tiff', classes_image)

# # # index = 1
# # # sorted_direc = sorted(os.listdir('gq-segmented-road'))
# # # sorted_direc_classes = sorted(os.listdir('gq-segmented-terrain-classes'))

# # # for i in range(0, len(sorted_direc), 16):
# # #     file_chunk = sorted_direc[i:i+16]
# # #     file_chunk_class = sorted_direc_classes[i:i+16]
# # #     create_stitched(None, file_chunk, file_chunk_class, index, ['gq-segmented-road', 'gq-segmented-road-classes'], 
# # #                     dest_direc=['images-stitched-road/', 'images-stitched-road-classes'])
# # #     index += 1

# used this before
path = '/home/anirud/Desktop/SemanticSeg/gq-segmented-road/'
classes_path = '/home/anirud/Desktop/SemanticSeg/gq-segmented-road-classes'
og_img = cv2.imread('/home/anirud/Desktop/SemanticSeg/alc-raw/gq-test.tiff')
#og_img = cv2.resize(og_img, (128, 128))

patch_size = 128
overlap = 0
#overlap = 0
# num_rows = 4 # prev 7
# num_cols = 4 # prev 7
num_rows = 4
num_cols = 4
total_patches = 16
#total_patches = 16

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
        raise Exception("implement this;")
    
    segmented_patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
    segmented_raw_patch = cv2.imread(classes_patch_path, cv2.IMREAD_UNCHANGED)


    start_row = row_idx * (patch_size - overlap)
    start_col = col_idx * (patch_size - overlap)
    end_row = start_row + patch_size
    end_col = start_col + patch_size
    #print('stitched_image[end_col]: ',stitched_image[end_col])
    stitched_image[start_row:end_row, start_col:end_col] = segmented_patch
    classes_image[start_row:end_row, start_col:end_col] = segmented_raw_patch


cv2.imwrite('OUTPUT/road_model.tiff', stitched_image)
cv2.imwrite('OUTPUT/road_model_classes.tiff', classes_image)
plt.figure(figsize=(12,8))
plt.subplot(231)
plt.imshow(og_img)
plt.subplot(232)
plt.imshow(stitched_image)
plt.subplot(233)
plt.imshow(classes_image)
plt.show()
    