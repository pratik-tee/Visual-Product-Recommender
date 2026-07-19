# ==========================================
# generate_embeddings.py
# ==========================================

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    SUBSET_CSV,
    IMAGE_FOLDER,
    EMBEDDING_FOLDER,
    BASELINE_EMBEDDING_FILE,
    FINETUNED_EMBEDDING_FILE,
    SIAMESE_EMBEDDING_FILE,
    METADATA_FILE,
)
from preprocess import preprocess_image
from feature_extractor import load_feature_extractor, extract_embedding

# ==========================================
# Create Embedding Folder
# ==========================================

os.makedirs(EMBEDDING_FOLDER, exist_ok=True)

# ==========================================
# Read Dataset 
# ==========================================

df = pd.read_csv(SUBSET_CSV)

df["image_path"] = df["id"].apply(
    lambda x: os.path.join(IMAGE_FOLDER, f"{x}.jpg")
)

df = df[df["image_path"].apply(os.path.exists)]

print(f"Total Images : {len(df)}")

# ==========================================
# Build metadata once — 
# ==========================================

metadata = []
for _, row in df.iterrows():
    metadata.append({
        "id": row["id"],
        "image_path": row["image_path"],
        "articleType": row["articleType"],
        "gender": row["gender"]
    })

with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata, f)

print("Metadata saved:", METADATA_FILE)

# ==========================================
# Generate embeddings for each model stage
# ==========================================

MODEL_STAGES = {
    "baseline": BASELINE_EMBEDDING_FILE,
    "finetuned": FINETUNED_EMBEDDING_FILE,
    "siamese": SIAMESE_EMBEDDING_FILE,
}

for model_type, output_path in MODEL_STAGES.items():

    print("\n" + "=" * 50)
    print(f"Generating embeddings for: {model_type}")
    print("=" * 50)

    model = load_feature_extractor(model_type=model_type)

    embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = preprocess_image(row["image_path"])
        embedding = extract_embedding(model, image)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    print(f"{model_type} embedding shape:", embeddings.shape)

    np.save(output_path, embeddings)

    print(f"Saved to: {output_path}")

print("\nAll embeddings generated successfully!")