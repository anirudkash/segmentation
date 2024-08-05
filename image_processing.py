import os
import cv2
import matplotlib.pyplot as plt

def create_patches(img_path, dest_direc=None, bgr=False, patch_size=256):
    if bgr:
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(img_path)
    h, w, _ = image.shape
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    patch_id = 1

    for top in range(0, h, patch_size):
        for left in range(0, w, patch_size):
            patch = image[top:top + patch_size, left:left + patch_size]

            if patch_id < 10:
                patch_str = str(0)+str(0)+str(patch_id)
            elif patch_id < 100:
                patch_str = str(0)+str(patch_id)
            else:
                patch_str = patch_id

            patch_fname = os.path.join(dest_direc, f"{name}_{patch_str}.tiff")
            cv2.imwrite(patch_fname, patch)
            patch_id += 1

def create_overlapping_patches(img_path, patch_size, overlap, dest_direc=None):

    image = cv2.imread(img_path)
    patches = []
    step = patch_size - overlap

    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    patch_id = 1

    for i in range(0, image.shape[0] - patch_size + 1, step):
        for j in range(0, image.shape[1] - patch_size + 1, step):
            patch = image[i:i+patch_size, j:j+patch_size]

            if patch_id < 10:
                patch_str = str(0)+str(0)+str(patch_id)
            elif patch_id < 100:
                patch_str = str(0)+str(patch_id)
            else:
                patch_str = patch_id
            
            patch_fname = os.path.join(dest_direc, f"{name}_{patch_str}.tiff")
            cv2.imwrite(patch_fname, patch)
            patch_id += 1

# Parameters
patch_size = 256
overlap = 128

def resize_images(img_path, dest_path, img_size=(1024, 1024), gray_scale=False):
    for fname in os.listdir(img_path):
        if gray_scale:
            test_img = cv2.imread(img_path + '/' + fname, 0)
        else:
            test_img = cv2.imread(img_path+'/'+fname, cv2.IMREAD_UNCHANGED)
        test_img = cv2.resize(test_img, img_size)
        
        filename = os.path.splitext(fname)[0]
        new_file = os.path.join(dest_path, filename+'.tiff')
        cv2.imwrite(new_file, test_img)
       # cv2.imwrite()


for fname in os.listdir('/home/anirud/Desktop/SemanticSeg/alc-raw'):
    img = os.path.join('/home/anirud/Desktop/SemanticSeg/alc-raw', fname)
    create_overlapping_patches(img, patch_size, overlap, dest_direc='/home/anirud/Desktop/SemanticSeg/alc-resized')


# train_img_path = '/home/anirud/Desktop/SemanticSeg/train-org-img'
# dest_train_img_path = '/home/anirud/Desktop/SemanticSeg/train-org-img-resized'


#resize_images('/home/anirud/Desktop/SemanticSeg/alc-raw/', '/home/anirud/Desktop/SemanticSeg/alc-resized')

# train_img_path = '/home/anirud/Desktop/SemanticSeg/train-org-img-resized'
# train_lab_path = '/home/anirud/Desktop/SemanticSeg/ColorMasks-TrainSet'
# test_img_path = '/home/anirud/Desktop/SemanticSeg/test-org-img-resized'
# test_lab_path = '/home/anirud/Desktop/SemanticSeg/ColorMasks-TestSet'
# train_img_path = '/home/anirud/Desktop/SemanticSeg/new-gq-patches'

# # # for fname in os.listdir('/home/anirud/Desktop/SemanticSeg/alc-raw'):
# # #         img = os.path.join('/home/anirud/Desktop/SemanticSeg/alc-raw', fname)
# # #         create_patches(img, dest_direc='/home/anirud/Desktop/SemanticSeg/alc-resized')

# # create_patches('/home/anirud/Desktop/SemanticSeg/gq-resized/gq.tiff', '/home/anirud/Desktop/SemanticSeg/gq-patches')

#resize_images('/home/anirud/Desktop/SemanticSeg/gq-raw', '/home/anirud/Desktop/SemanticSeg/gq-resized', img_size=(4096, 4096))


#resize_images(test_img_path, '/home/anirud/Desktop/SemanticSeg/test-org-img-resized')

#resize_images('/home/anirud/Desktop/SemanticSeg/gq_rawdata', '/home/anirud/Desktop/SemanticSeg/gq_rawdata')

#img = cv2.imread('/home/anirud/Desktop/SemanticSeg/GQmashup_R1C1 (1).tiff')
#create_patches(img_path='/home/anirud/Desktop/SemanticSeg/gq-image.png', dest_direc='/home/anirud/Desktop/SemanticSeg/gq-patches')



# for fname in os.listdir(train_lab_path):
#     if counter <= 4500:
#         img = os.path.join(train_lab_path, fname)
#         create_patches(img, dest_direc='/home/anirud/Desktop/SemanticSeg/TrainImgsOne(git)')
#         counter += 1
#         print(fname)
