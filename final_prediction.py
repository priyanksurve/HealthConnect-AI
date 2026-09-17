import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ==========================================
# 1. Load Models
# ==========================================

print("Loading U-Net model...")

unet_model = tf.keras.models.load_model(
    "unet_lung_segmentation.keras"
)

print("Loading Pneumonia Classifier...")

classifier_model = tf.keras.models.load_model(
    "pneumonia_classifier.keras"
)

print("Models loaded successfully!")


# ==========================================
# 2. Input X-ray Image
# ==========================================

IMAGE_PATH = r"D:\minorproject\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg"


# ==========================================
# 3. Read Original Image
# ==========================================

original_image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if original_image is None:

    print("Image not found!")

    exit()


# ==========================================
# 4. U-Net Lung Segmentation
# ==========================================

print("Predicting lung segmentation...")


# U-Net expects 256 x 256 x 1

unet_image = cv2.resize(
    original_image,
    (256, 256)
)


unet_image = unet_image / 255.0


unet_input = np.expand_dims(
    unet_image,
    axis=0
)


unet_input = np.expand_dims(
    unet_input,
    axis=-1
)


# Prediction

predicted_mask = unet_model.predict(
    unet_input
)[0]


# Convert probability mask into binary mask

predicted_mask = (
    predicted_mask > 0.5
).astype(np.uint8)


# Remove channel dimension

predicted_mask = predicted_mask[:, :, 0]


# ==========================================
# 5. Prepare Image for Classifier
# ==========================================

print("Preparing image for classification...")


# Classifier expects:
# 224 x 224 x 3

classifier_image = cv2.imread(
    IMAGE_PATH
)


classifier_image = cv2.resize(
    classifier_image,
    (224, 224)
)


classifier_image = classifier_image / 255.0


# Shape becomes:
# (1, 224, 224, 3)

classifier_input = np.expand_dims(
    classifier_image,
    axis=0
)


# ==========================================
# 6. Pneumonia Prediction
# ==========================================

print("Predicting pneumonia...")


prediction = classifier_model.predict(
    classifier_input
)[0][0]


# ==========================================
# 7. Final Result
# ==========================================

if prediction > 0.5:

    result = "PNEUMONIA"

    confidence = prediction * 100

else:

    result = "NORMAL"

    confidence = (1 - prediction) * 100


print("\n")
print("====================================")
print("        FINAL PREDICTION")
print("====================================")

print("Result:", result)

print("Confidence:", round(confidence, 2), "%")

print("====================================")


# ==========================================
# 8. Display Results
# ==========================================

plt.figure(
    figsize=(15, 5)
)


# ------------------------------------------
# Original X-ray
# ------------------------------------------

plt.subplot(
    1,
    3,
    1
)


plt.imshow(
    original_image,
    cmap="gray"
)


plt.title(
    "Original X-ray"
)


plt.axis(
    "off"
)


# ------------------------------------------
# Predicted Lung Mask
# ------------------------------------------

plt.subplot(
    1,
    3,
    2
)


plt.imshow(
    predicted_mask,
    cmap="gray"
)


plt.title(
    "Predicted Lung Mask"
)


plt.axis(
    "off"
)


# ------------------------------------------
# Final Prediction
# ------------------------------------------

plt.subplot(
    1,
    3,
    3
)


plt.imshow(
    original_image,
    cmap="gray"
)


plt.title(
    f"{result}\nConfidence: {confidence:.2f}%"
)


plt.axis(
    "off"
)


plt.tight_layout()


plt.show()