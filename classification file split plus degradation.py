import os
import cv2
import numpy as np
import random

# folder containing ALL images
source_dir = "/Users/maisycook/Documents/Plant Disease Detection/PlantVillage_for_object_detection/Dataset/images"

# output YOLO dataset
output_dir = "/Users/maisycook/Documents/Plant Disease Detection/degradedclassificationdataset"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

def degrade_image(img):

    h, w = img.shape[:2]

    # slight blur (focus softness common in older phones)
    if random.random() < 0.3:
        k = random.choice([3,5])
        img = cv2.GaussianBlur(img, (k,k), 0.5)

    # mild sensor noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 6, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # slight resolution loss
    if random.random() < 0.4:
        scale = random.uniform(0.7, 0.9)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # moderate jpeg compression
    if random.random() < 0.4:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(65, 85)]
        _, encimg = cv2.imencode(".jpg", img, encode_param)
        img = cv2.imdecode(encimg, 1)

    # slight brightness variation
    if random.random() < 0.25:
        alpha = random.uniform(0.9, 1.1)
        beta = random.randint(-8, 8)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img


# collect images by class
class_images = {}

for file in os.listdir(source_dir):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # class name from filename
    class_name = file.split("_")[0]

    if class_name not in class_images:
        class_images[class_name] = []

    class_images[class_name].append(file)


# process each class
for class_name, files in class_images.items():

    random.shuffle(files)

    total = len(files)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, split_files in splits.items():

        class_folder = os.path.join(output_dir, split, class_name)
        os.makedirs(class_folder, exist_ok=True)

        for file in split_files:

            src = os.path.join(source_dir, file)
            dst = os.path.join(class_folder, file)

            img = cv2.imread(src)

            # degrade ~70% of images
            if random.random() < 0.7:
                img = degrade_image(img)

            cv2.imwrite(dst, img)


print("Dataset split and degraded.")