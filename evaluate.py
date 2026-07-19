# ==========================================
# evaluate.py
# ==========================================
#


import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    METADATA_FILE,
    BASELINE_EMBEDDING_FILE,
    FINETUNED_EMBEDDING_FILE,
    SIAMESE_EMBEDDING_FILE,
    TOP_K,
)

# ==========================================
# Load metadata (shared across all 3 models)
# ==========================================

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

df_meta = pd.DataFrame(metadata)

MODELS = {
    "baseline": BASELINE_EMBEDDING_FILE,
    "finetuned": FINETUNED_EMBEDDING_FILE,
    "siamese": SIAMESE_EMBEDDING_FILE,
}

# ==========================================
# Precision@K / Recall@K
# ==========================================

def precision_recall_at_k(embeddings, labels, k=TOP_K, n_queries=200, seed=42):
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    query_indices = rng.choice(n, size=min(n_queries, n), replace=False)

    sims = cosine_similarity(embeddings)

    precisions = []
    recalls = []

    labels = np.array(labels)

    for q_idx in query_indices:

        query_label = labels[q_idx]

        total_relevant = np.sum(labels == query_label) - 1
        if total_relevant <= 0:
            continue

        sim_scores = sims[q_idx].copy()
        sim_scores[q_idx] = -1

        top_k_idx = np.argsort(sim_scores)[::-1][:k]
        top_k_labels = labels[top_k_idx]

        relevant_in_topk = np.sum(top_k_labels == query_label)

        precision = relevant_in_topk / k
        recall = relevant_in_topk / total_relevant

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)


# ==========================================
# Timing
# ==========================================

def measure_embedding_generation_time(embedding_path):
    start = time.time()
    _ = np.load(embedding_path)
    return time.time() - start


def measure_query_time(embeddings, k=TOP_K, n_trials=50):
    rng = np.random.default_rng(0)
    n = len(embeddings)
    times = []

    for _ in range(n_trials):
        idx = rng.integers(0, n)
        query_vec = embeddings[idx:idx + 1]

        start = time.time()
        sims = cosine_similarity(query_vec, embeddings)[0]
        _ = np.argsort(sims)[::-1][:k]
        times.append(time.time() - start)

    return np.mean(times)


# ==========================================
# Run Evaluation for All 3 Models
# ==========================================

results = {}

for model_name, emb_path in MODELS.items():

    print(f"\nEvaluating: {model_name}")

    embeddings = np.load(emb_path)
    labels = df_meta["articleType"].values

    precision, recall = precision_recall_at_k(embeddings, labels, k=TOP_K)
    embed_load_time = measure_embedding_generation_time(emb_path)
    query_time = measure_query_time(embeddings, k=TOP_K)

    results[model_name] = {
        "precision@k": precision,
        "recall@k": recall,
        "embedding_load_time_sec": embed_load_time,
        "avg_query_time_sec": query_time,
    }

    print(f"  Precision@{TOP_K}: {precision:.4f}")
    print(f"  Recall@{TOP_K}:    {recall:.4f}")
    print(f"  Avg query time:   {query_time * 1000:.2f} ms")

# ==========================================
# Print Summary Table
# ==========================================

results_df = pd.DataFrame(results).T
print("\n" + "=" * 60)
print("SUMMARY: Baseline vs Fine-tuned vs Siamese")
print("=" * 60)
print(results_df)

results_df.to_csv("evaluation_results.csv")
print("\nSaved to evaluation_results.csv")

# ==========================================
# Qualitative Visual Comparison
# ==========================================

def show_comparison(query_idx, k=TOP_K):

    fig, axes = plt.subplots(3, k + 1, figsize=(3 * (k + 1), 9))

    query_path = df_meta.iloc[query_idx]["image_path"]
    query_label = df_meta.iloc[query_idx]["articleType"]

    for row, (model_name, emb_path) in enumerate(MODELS.items()):

        embeddings = np.load(emb_path)
        sims = cosine_similarity(
            embeddings[query_idx:query_idx + 1], embeddings
        )[0]
        sims[query_idx] = -1
        top_k_idx = np.argsort(sims)[::-1][:k]

        axes[row, 0].imshow(plt.imread(query_path))
        axes[row, 0].set_title(f"Query\n({model_name})", fontsize=9)
        axes[row, 0].axis("off")

        for col, idx in enumerate(top_k_idx):
            img_path = df_meta.iloc[idx]["image_path"]
            label = df_meta.iloc[idx]["articleType"]
            score = sims[idx]

            axes[row, col + 1].imshow(plt.imread(img_path))
            axes[row, col + 1].set_title(f"{label}\n{score:.2f}", fontsize=8)
            axes[row, col + 1].axis("off")

    plt.suptitle(f"Query category: {query_label}", fontsize=12)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    print("\nSaved visual comparison to model_comparison.png")
    plt.show()


if __name__ == "__main__":
    show_comparison(query_idx=0, k=TOP_K)