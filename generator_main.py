import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from unet_model import UNet, jacard_coef
from keras.utils import normalize
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# class_colors = [
#     (0, 0, 0), # background
#     (255, 0, 0), # building flooded
#     (180, 120, 120),  # building non-flooded --
#     (160, 150, 20), # road-flooded
#     (140, 140, 140),   # road-non-flooded
#     (61, 230, 250),    # water --
#     (0, 82, 255),      # tree
#     (255, 0, 245),     # vehicle
#     (255, 235, 0), # pool
#     (4, 250, 7)        # grass
# ]

class_colors = [
    (0, 0, 0), # background
    (41, 25, 222), # building non-flooded
    (200, 191, 231), # building flooded
    (52, 53, 54), # road non-flooded
    (0, 57, 231), # road flooded
    (0, 162, 232), # water
    (202, 216, 231), # pool
    (231, 37, 150), # vehicle
    (18, 92, 39), # tree
    (37, 191, 82) # grass
]

background = np.array(class_colors[0])
building_nonflooded = np.array(class_colors[1])
building_flooded = np.array(class_colors[2])
road_nonflooded = np.array(class_colors[3])
road_flooded = np.array(class_colors[4])
water = np.array(class_colors[5])
pool = np.array(class_colors[6])
vehicle = np.array(class_colors[7])
tree = np.array(class_colors[8])
grass = np.array(class_colors[9])

# background = np.array(class_colors[0])
# building_flooded = np.array(class_colors[1])
# building_nonflooded = np.array(class_colors[2])
# road_flooded = np.array(class_colors[2])
# road_nonflooded = np.array(class_colors[3])
# water = np.array(class_colors[4])
# tree = np.array(class_colors[5])
# vehicle = np.array(class_colors[6])
# pool = np.array(class_colors[7])
# grass = np.array(class_colors[9])

num_classes = len(class_colors)

# train_imgs_direc = '/home/anirud/Desktop/SemanticSeg/train-org-img-patches'
# train_imgs_labs_direc = '/home/anirud/Desktop/SemanticSeg/ColorMasks-TrainSet-patches'

# train_imgs_direc = '/home/anirud/Desktop/SemanticSeg/TrainImgsOne'
# train_imgs_labs_direc = '/home/anirud/Desktop/SemanticSeg/MaskPatchesOne'

# this is the one I've been using all of this time before using my own gq image
# # # train_imgs_direc = 'TrainImgsOne'
# # # train_imgs_labs_direc = 'MaskPatchesOne'

train_imgs_direc = 'training-images'
train_imgs_labs_direc = 'training-images-labeled'

# USE THE ONE BELOW FOR BUILDINGS/TERRAIN MODEL
# # train_imgs_direc = '/home/anirud/Desktop/SemanticSeg/TrainImgsOneWithBuilds'
# # train_imgs_labs_direc = '/home/anirud/Desktop/SemanticSeg/MaskPatchesOneWithBuilds'

#train_imgs_labs_direc = '/home/anirud/Desktop/SemanticSeg/terrain_model/terrain_only'

# train_imgs_direc = '/home/anirud/Desktop/SemanticSeg/train-org-img'
# train_imgs_labs_direc = '/home/anirud/Desktop/SemanticSeg/terrain_model/terrain_only'

def load_images(img_direc, gt_direc):

    train_imgs = []
    train_imgs_labs = []

    train_imgs_entries = os.listdir(img_direc)
    train_imgs_entries.sort()

    train_imgs_labs_entries = os.listdir(gt_direc)
    train_imgs_labs_entries.sort()

    for fname in train_imgs_entries:
        path = os.path.join(img_direc, fname) 
        # might have to read in the image as gray scale 
        img = cv2.imread(path)
        img = cv2.resize(img, (128,128))
        train_imgs.append(img)

    for fname in train_imgs_labs_entries:
        path = os.path.join(gt_direc, fname)
        img = cv2.imread(path)
        img = cv2.resize(img, (128,128))
       # img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        train_imgs_labs.append(img)

    train_images = np.array(train_imgs)
    train_masks = np.array(train_imgs_labs)

    return train_images, train_masks

train_images, train_masks = load_images(train_imgs_direc, train_imgs_labs_direc)
print('Loaded images')
new_train_masks = np.empty_like(train_masks)

for i in range(train_masks.shape[0]):
    new_train_masks[i] = cv2.cvtColor(train_masks[i], cv2.COLOR_BGR2RGB)

train_masks = new_train_masks

def rgb_to_2D_label(label):
    label_seg = np.zeros(label.shape,dtype=np.uint8)

    label_seg[np.all(label==background, axis=-1)] = 0
    label_seg[np.all(label==building_nonflooded, axis=-1)] = 1
    label_seg[np.all(label==building_flooded, axis=-1)] = 2
    label_seg[np.all(label==road_nonflooded, axis=-1)] = 3
    label_seg[np.all(label==road_flooded, axis=-1)] = 4
    label_seg[np.all(label==water, axis=-1)] = 5
    label_seg[np.all(label==pool, axis=-1)] = 6
    label_seg[np.all(label==vehicle, axis=-1)] = 7
    label_seg[np.all(label==tree, axis=-1)] = 8
    label_seg[np.all(label==grass, axis=-1)] = 9

    # label_seg[np.all(label==background, axis=-1)] = 0
    # label_seg[np.all(label==building_flooded, axis=-1)] = 1
    # label_seg[np.all(label==building_nonflooded, axis=-1)] = 2
    # label_seg[np.all(label==road_flooded, axis=-1)] = 3
    # label_seg[np.all(label==road_nonflooded, axis=-1)] = 4
    # label_seg[np.all(label==water, axis=-1)] = 5
    # label_seg[np.all(label==tree, axis=-1)] = 6
    # label_seg[np.all(label==vehicle, axis=-1)] = 7
    # label_seg[np.all(label==pool, axis=-1)] = 8
    # label_seg[np.all(label==grass, axis=-1)] = 9

    label_seg = label_seg[:,:,0]

    return label_seg

labels = []
for i in range(train_masks.shape[0]):
    label = rgb_to_2D_label(train_masks[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)
print('Labeling done')

labels_cat = to_categorical(labels, num_classes=len(class_colors))
print('One hot encoding performed')
X_train, X_test, y_train, y_test = train_test_split(train_images, labels_cat, test_size=0.2, random_state=42)

print('Split the dataset')

h = X_train.shape[1]
w = X_train.shape[2]
c = X_train.shape[3]

model = UNet(num_classes=len(class_colors), input_size=(h,w,c))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
weights = model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=150, validation_data=(X_test, y_test), shuffle=False)
print(f'Model trained on {4} epochs and a batch size of {16}')
model.save_weights('terrain_new.weights.h5')
y_test_argmax = np.argmax(y_test, axis=3)

test_img_number = random.randint(0, len(X_test))
print('test_img_number index: ', test_img_number)
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0,:,:]

# plt.figure(figsize=(12,8))
# plt.subplot(231)
# plt.title('Test Image')
# plt.imshow(test_img)
# plt.subplot(232)
# plt.title('Test Label')
# plt.imshow(ground_truth, cmap='jet')
# plt.subplot(233)
# plt.title('Predicted Image')
# plt.imshow(predicted_img, cmap='jet')
# plt.show()

try:
    while True:
        test_img_number = random.randint(0, len(X_test))
        print('test_img_number index: ', test_img_number)
        test_img = X_test[test_img_number]
        ground_truth = y_test_argmax[test_img_number]
        test_img_input = np.expand_dims(test_img, 0)
        prediction = (model.predict(test_img_input))
        predicted_img = np.argmax(prediction, axis=3)[0,:,:]
        #print('Classes I found: ', [class_labs[idx] for idx in np.unique(predicted_img)])
        plt.figure(figsize=(12,8))
        plt.subplot(231)
        plt.title('Test Image')
        plt.imshow(test_img)
        plt.subplot(232)
        plt.title('Groudn Truth')
        plt.imshow(ground_truth)
        plt.subplot(233)
        plt.title('Generated Image (L_generated)')
        plt.imshow(predicted_img)
        plt.show()
        print('Done')
        
except KeyboardInterrupt:
    pass
