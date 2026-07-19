# ==========================================
# preprocess.py
# ==========================================

import cv2
import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input

from config import IMAGE_SIZE


def preprocess_image(image_path):
    """
    Read and preprocess an image for ResNet50.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert to float
    image = image.astype(np.float32)

    # ResNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image