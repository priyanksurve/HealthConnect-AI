import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# =========================
# 1. Dataset Paths
# =========================

dataset_path = r"D:\minorproject\archive\Lung Segmentation"

image_path = os.path.join(dataset_path, "CXR_png")
mask_path = os.path.join(dataset_path, "masks")


# =========================
# 2. Output Paths
# =========================

output_path = r"D:\minorproject\processed_dataset"

train_images_path = os.path.join(output_path, "train", "images")
train_masks_path = os.path.join(output_path, "train", "masks")

val_images_path = os.path.join(output_path, "val", "images")
val_masks_path = os.path.join(output_path, "val", "masks")


# Create folders
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_masks_path, exist_ok=True)

os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_masks_path, exist_ok=True)


# =========================
# 3. Find Matching Pairs
# =========================

images = [
    file for file in os.listdir(image_path)
    if file.endswith(".png")
]

masks = [
    file for file in os.listdir(mask_path)
    if file.endswith(".png")
]


matching_pairs = []

for image in images:

    mask_name = image.replace(".png", "_mask.png")

    if mask_name in masks:
        matching_pairs.append((image, mask_name))


print("Total Matching Pairs:", len(matching_pairs))


# =========================
# 4. Train Validation Split
# =========================

train_pairs, val_pairs = train_test_split(
    matching_pairs,
    test_size=0.2,
    random_state=42
)


print("Training Images:", len(train_pairs))
print("Validation Images:", len(val_pairs))


# =========================
# 5. Image Preprocessing
# =========================

IMG_SIZE = (256, 256)


def process_and_save(pairs, image_output, mask_output):

    for image_name, mask_name in pairs:

        # Read X-ray as grayscale
        image = cv2.imread(
            os.path.join(image_path, image_name),
            cv2.IMREAD_GRAYSCALE
        )

        # Read mask as grayscale
        mask = cv2.imread(
            os.path.join(mask_path, mask_name),
            cv2.IMREAD_GRAYSCALE
        )

        # Resize image
        image = cv2.resize(
            image,
            IMG_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # Resize mask
        mask = cv2.resize(
            mask,
            IMG_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize image from 0-255 to 0-1
        image = image / 255.0

        # Convert mask into binary mask
        mask = (mask > 127).astype(np.uint8) * 255

        # Save processed image
        image_save_path = os.path.join(
            image_output,
            image_name
        )

        mask_save_path = os.path.join(
            mask_output,
            mask_name
        )

        # Convert normalized image back to 0-255 for saving
        image_to_save = (image * 255).astype(np.uint8)

        cv2.imwrite(image_save_path, image_to_save)
        cv2.imwrite(mask_save_path, mask)


# =========================
# 6. Process Dataset
# =========================

print("\nProcessing training data...")

process_and_save(
    train_pairs,
    train_images_path,
    train_masks_path
)


print("Processing validation data...")

process_and_save(
    val_pairs,
    val_images_path,
    val_masks_path
)


print("\nPreprocessing Completed Successfully!")