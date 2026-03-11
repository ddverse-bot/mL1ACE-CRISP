import os
import shutil
import random

image_dir = "data/camus/images"
mask_dir = "data/camus/masks"

train_img = "data/train/images"
train_mask = "data/train/masks"

test_img = "data/test/images"
test_mask = "data/test/masks"

os.makedirs(train_img, exist_ok=True)
os.makedirs(train_mask, exist_ok=True)
os.makedirs(test_img, exist_ok=True)
os.makedirs(test_mask, exist_ok=True)

images = os.listdir(image_dir)
random.shuffle(images)

split = int(0.8 * len(images))

train_files = images[:split]
test_files = images[split:]


def copy_files(file_list, img_dst, mask_dst):

    for img in file_list:

        mask = img.replace("frame", "mask")

        img_path = os.path.join(image_dir, img)
        mask_path = os.path.join(mask_dir, mask)

        if os.path.exists(img_path) and os.path.exists(mask_path):

            shutil.copy(img_path, img_dst)
            shutil.copy(mask_path, mask_dst)


copy_files(train_files, train_img, train_mask)
copy_files(test_files, test_img, test_mask)

print("Dataset split completed")