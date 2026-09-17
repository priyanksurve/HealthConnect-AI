import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# 1. Dataset Paths
# ==============================

train_dir = "chest_xray/train"
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

# ==============================
# 2. Image Configuration
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# 3. Data Augmentation
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==============================
# 4. Load Dataset
# ==============================

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class Labels:", train_data.class_indices)

# ==============================
# 5. CNN Model
# ==============================

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# ==============================
# 6. Compile Model
# ==============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# 7. Train Model
# ==============================

print("\nStarting Pneumonia Classification Training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ==============================
# 8. Evaluate Model
# ==============================

print("\nEvaluating Model on Test Data...\n")

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy * 100, "%")

# ==============================
# 9. Save Model
# ==============================

model.save("pneumonia_classifier.keras")

print("\nModel saved successfully!")