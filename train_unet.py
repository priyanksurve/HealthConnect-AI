import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model


# =========================
# 1. Dataset Paths
# =========================

train_images_path = r"D:\minorproject\processed_dataset\train\images"
train_masks_path = r"D:\minorproject\processed_dataset\train\masks"

val_images_path = r"D:\minorproject\processed_dataset\val\images"
val_masks_path = r"D:\minorproject\processed_dataset\val\masks"


# =========================
# 2. Settings
# =========================

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20


# =========================
# 3. Load Dataset
# =========================

def load_data(image_path, mask_path):

    images = []
    masks = []

    image_files = sorted(os.listdir(image_path))

    for image_name in image_files:

        image_file = os.path.join(image_path, image_name)

        # Corresponding mask name
        mask_name = image_name.replace(
            ".png",
            "_mask.png"
        )

        mask_file = os.path.join(
            mask_path,
            mask_name
        )

        # Check mask exists
        if not os.path.exists(mask_file):
            continue

        # Read image
        image = cv2.imread(
            image_file,
            cv2.IMREAD_GRAYSCALE
        )

        # Read mask
        mask = cv2.imread(
            mask_file,
            cv2.IMREAD_GRAYSCALE
        )

        # Resize
        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE)
        )

        # Normalize image
        image = image / 255.0

        # Convert mask to binary
        mask = mask / 255.0

        # Add channel dimension
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)


# =========================
# 4. Load Training Data
# =========================

print("Loading training data...")

X_train, Y_train = load_data(
    train_images_path,
    train_masks_path
)

print("Training Images:", X_train.shape)
print("Training Masks:", Y_train.shape)


# =========================
# 5. Load Validation Data
# =========================

print("\nLoading validation data...")

X_val, Y_val = load_data(
    val_images_path,
    val_masks_path
)

print("Validation Images:", X_val.shape)
print("Validation Masks:", Y_val.shape)


# =========================
# 6. U-Net Model
# =========================

def build_unet():

    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 1)
    )


    # Encoder Block 1
    c1 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    c1 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(c1)

    p1 = layers.MaxPooling2D(
        (2, 2)
    )(c1)


    # Encoder Block 2
    c2 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(p1)

    c2 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(c2)

    p2 = layers.MaxPooling2D(
        (2, 2)
    )(c2)


    # Encoder Block 3
    c3 = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(p2)

    c3 = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(c3)

    p3 = layers.MaxPooling2D(
        (2, 2)
    )(c3)


    # Bottleneck
    c4 = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    )(p3)

    c4 = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    )(c4)


    # Decoder Block 1
    u5 = layers.UpSampling2D(
        (2, 2)
    )(c4)

    u5 = layers.concatenate(
        [u5, c3]
    )

    c5 = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(u5)

    c5 = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(c5)


    # Decoder Block 2
    u6 = layers.UpSampling2D(
        (2, 2)
    )(c5)

    u6 = layers.concatenate(
        [u6, c2]
    )

    c6 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(u6)

    c6 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(c6)


    # Decoder Block 3
    u7 = layers.UpSampling2D(
        (2, 2)
    )(c6)

    u7 = layers.concatenate(
        [u7, c1]
    )

    c7 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(u7)

    c7 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(c7)


    # Output Layer
    outputs = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid"
    )(c7)


    model = Model(
        inputs,
        outputs
    )

    return model


# =========================
# 7. Create Model
# =========================

model = build_unet()


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# =========================
# 8. Train Model
# =========================

print("\nStarting U-Net Training...")


history = model.fit(
    X_train,
    Y_train,

    validation_data=(
        X_val,
        Y_val
    ),

    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)


# =========================
# 9. Save Model
# =========================

model.save(
    "unet_lung_segmentation.keras"
)


print("\nTraining Completed Successfully!")

print(
    "Model saved as: "
    "unet_lung_segmentation.keras"
)