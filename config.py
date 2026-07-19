# ==========================================
# config.py
# ==========================================

import os

# ==========================================
# Project Root
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Dataset Paths
# ==========================================

DATASET_PATH = os.path.join(BASE_DIR, "dataset")

IMAGE_FOLDER = os.path.join(DATASET_PATH, "images")

STYLE_CSV = os.path.join(DATASET_PATH, "styles.csv")

SUBSET_CSV = os.path.join(DATASET_PATH, "subset.csv")

PAIR_CSV = os.path.join(DATASET_PATH, "pairs.csv")

# ==========================================
# Categories 
# ==========================================

CATEGORIES = [
    "Shirts",
    "Tshirts",
    "Jeans",
    "Dresses",
    "Sandals",
    "Handbags",
    "Watches",
]

IMAGES_PER_CATEGORY = 250

# ==========================================
# Model Paths
# ==========================================

MODEL_FOLDER = os.path.join(BASE_DIR, "models")

TRANSFER_MODEL = os.path.join(
    MODEL_FOLDER,
    "resnet50_finetuned.keras"
)

SIAMESE_MODEL = os.path.join(
    MODEL_FOLDER,
    "siamese_model.keras"
)

SIAMESE_FINAL_MODEL = os.path.join(
    MODEL_FOLDER,
    "siamese_model_final.keras"
)

# ==========================================
# Embedding Paths
# ==========================================

EMBEDDING_FOLDER = os.path.join(BASE_DIR, "embeddings")

BASELINE_EMBEDDING_FILE = os.path.join(
    EMBEDDING_FOLDER,
    "baseline_embeddings.npy"
)

FINETUNED_EMBEDDING_FILE = os.path.join(
    EMBEDDING_FOLDER,
    "finetuned_embeddings.npy"
)

SIAMESE_EMBEDDING_FILE = os.path.join(
    EMBEDDING_FOLDER,
    "siamese_embeddings.npy"
)


METADATA_FILE = os.path.join(
    EMBEDDING_FOLDER,
    "metadata.pkl"
)

# ==========================================
# Image Settings
# ==========================================

IMAGE_SIZE = (224, 224)

IMAGE_CHANNELS = 3

# ==========================================
# Embedding Settings
# ==========================================

EMBEDDING_DIM = 512

TOP_K = 5

# ==========================================
# Training Settings
# ==========================================

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 1e-4

RANDOM_STATE = 42