import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# =========================
# 1. Paths
# =========================

MODEL_PATH = r"D:\minorproject\unet_lung_segmentation.keras"

IMAGE_PATH = r"D:\minorproject\processed_dataset\val\images"

MASK_PATH = r"D:\minorproject\processed_dataset\val\masks"


# =========================
# 2. Load Model
# =========================

print("Loading U-Net model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# =========================
# 3. Select an Image
# =========================

image_files = [
    file for file in os.listdir(IMAGE_PATH)
    if file.endswith(".png")
]


# Select first image
image_name = image_files[0]

mask_name = image_name.replace(
    ".png",
    "_mask.png"
)


image_file = os.path.join(
    IMAGE_PATH,
    image_name
)

mask_file = os.path.join(
    MASK_PATH,
    mask_name
)


# =========================
# 4. Load Image and Mask
# =========================

image = cv2.imread(
    image_file,
    cv2.IMREAD_GRAYSCALE
)

actual_mask = cv2.imread(
    mask_file,
    cv2.IMREAD_GRAYSCALE
)


# =========================
# 5. Preprocess Image
# =========================

input_image = image / 255.0

input_image = np.expand_dims(
    input_image,
    axis=-1
)

input_image = np.expand_dims(
    input_image,
    axis=0
)


# =========================
# 6. Predict Mask
# =========================

print("Predicting lung segmentation...")

prediction = model.predict(
    input_image
)


# Remove batch and channel dimensions
predicted_mask = prediction[0, :, :, 0]


# Convert probability to binary mask
predicted_mask = (
    predicted_mask > 0.5
).astype(np.uint8) * 255


# =========================
# 7. Display Results
# =========================

plt.figure(figsize=(15, 5))


# Original X-ray
plt.subplot(1, 3, 1)

plt.imshow(
    image,
    cmap="gray"
)

plt.title("Original X-ray")

plt.axis("off")


# Actual Mask
plt.subplot(1, 3, 2)

plt.imshow(
    actual_mask,
    cmap="gray"
)

plt.title("Actual Lung Mask")

plt.axis("off")


# Predicted Mask
plt.subplot(1, 3, 3)

plt.imshow(
    predicted_mask,
    cmap="gray"
)

plt.title("Predicted Lung Mask")

plt.axis("off")


plt.tight_layout()

plt.show()

plt.tight_layout()

plt.savefig(
    r"D:\minorproject\segmentation_result.png"
)

plt.show()