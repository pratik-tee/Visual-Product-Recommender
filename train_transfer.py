# ==========================================
# train_transfer.py
# ==========================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from config import *

# ------------------------------------
# Read Dataset
# ------------------------------------

df = pd.read_csv(SUBSET_CSV)

df["image_path"] = df["id"].apply(
    lambda x: os.path.join(IMAGE_FOLDER, f"{x}.jpg")
)

df = df[df["image_path"].apply(os.path.exists)]

print("Total Images :", len(df))

# ------------------------------------
# Split Dataset
# ------------------------------------

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["articleType"]
)

# ------------------------------------
# Image Generators
# ------------------------------------

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="articleType",
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="articleType",
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode="categorical"
)

# ------------------------------------
# Load ResNet50
# ------------------------------------

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

# ------------------------------------
# Classification Head
# ------------------------------------

x = GlobalAveragePooling2D()(base_model.output)

x = Dense(512, activation="relu")(x)

x = Dropout(0.3)(x)

num_classes = len(train_generator.class_indices)

predictions = Dense(
    num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# ------------------------------------
# Compile
# ------------------------------------

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------
# Save Best Model
# ------------------------------------

os.makedirs(MODEL_FOLDER, exist_ok=True)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_FOLDER, "resnet50_finetuned.keras"),
    monitor="val_accuracy",
    save_best_only=True
)

# ------------------------------------
# Train
# ------------------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[checkpoint]
)

print("Training Completed")