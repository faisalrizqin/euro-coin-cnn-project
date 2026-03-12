import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm

# dataset asli
source_dir = "dataset/coins-dataset/classified/train"

# dataset baru (balanced)
target_dir = "dataset_balanced/train"

target_per_class = 1000

os.makedirs(target_dir, exist_ok = True)

classes = os.listdir(source_dir)

def augment_image(img):

    # rotation
    angle = random.uniform(-25, 25)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # brightness
    value = random.randint(-40, 40)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    v = hsv[:, :, 2].astype(np.int16)
    v = np.clip(v + value, 0, 255)

    hsv[:, :, 2] = v.astype(np.uint8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

for cls in classes:

    print("Processing: ", cls)

    class_src = os.path.join(source_dir, cls)
    class_dst = os.path.join(target_dir, cls)

    os.makedirs(class_dst, exist_ok = True)

    images = os.listdir(class_src)

    # copy original images
    for img_name in images:
        src = os.path.join(class_src, img_name)
        dst = os.path.join(class_dst, img_name)
        shutil.copy(src, dst)

    count = len(images)

    while count < target_per_class:

        img_name = random.choice(images)
        img_path = os.path.join(class_src, img_name)

        img = cv2.imread(img_path)

        aug = augment_image(img)

        save_name = f"aug_{count}.jpg"
        save_path = os.path.join(class_dst, save_name)

        cv2.imwrite(save_path, aug)

        count += 1

    # jika lebih dari 1000, random ambil 1000
    files = os.listdir(class_dst)
    if len(files) > target_per_class:

        remove = random.sample(files, len(files) - target_per_class)

        for r in remove:
            os.remove(os.path.join(class_dst, r))

print("Dataset balanced selesai!")