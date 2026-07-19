# 👕 Visual Product Recommendation System

An image-based recommendation system that retrieves visually similar fashion products using **Transfer Learning (ResNet50)**, **Siamese Networks**, and **Cosine Similarity**.

---

## 📌 Problem Statement

Traditional keyword-based product search cannot effectively capture visual similarities such as color, style, pattern, or texture.

This project builds a deep learning-based visual recommendation engine that accepts an input image and retrieves visually similar fashion products.

---

# Features

- Image Upload using Streamlit
- Transfer Learning using ResNet50
- Siamese Network for similarity learning
- Baseline, Fine-tuned and Siamese Embeddings
- Cosine Similarity based Retrieval
- Precomputed Embeddings for Faster Search
- Precision@K & Recall@K Evaluation
- Interactive Recommendation UI

---

# Project Structure

```
Visual Product Recommendation System
│
├── dataset/
│ ├── images/
│ ├── styles.csv
│ ├── subset.csv
│ └── pairs.csv
│
├── embeddings/
│ ├── baseline_embeddings.npy
│ ├── finetuned_embeddings.npy
│ ├── siamese_embeddings.npy
│ └── metadata.pkl
│
├── models/
│ ├── resnet50_finetuned.keras
│ └── siamese_model_final.keras
│
├── app.py
├── config.py
├── preprocess.py
├── feature_extractor.py
├── recommendation.py
├── generate_embeddings.py
├── train_transfer.py
├── train_siamese.py
├── create_pairs.py
├── create_subset.py
├── pair_generator.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

# Workflow

```
Dataset
      │
      ▼
Subset Creation
      │
      ▼
Preprocessing
      │
      ▼
Transfer Learning (ResNet50)
      │
      ▼
Embedding Generation
      │
      ▼
Baseline Similarity
      │
      ▼
Siamese Network Training
      │
      ▼
Improved Embeddings
      │
      ▼
Cosine Similarity Search
      │
      ▼
Top-K Recommendation
      │
      ▼
Streamlit UI
```

---

# Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Streamlit

---

# Evaluation

The recommendation system is evaluated using:

- Precision@K
- Recall@K
- Visual Comparison
- Embedding Generation Time
- Inference Time

---

# Installation

```bash
git clone https://github.com/yourusername/Visual-Product-Recommendation-System.git

cd Visual-Product-Recommendation-System

pip install -r requirements.txt
```

---

# Run

```bash
streamlit run app.py
```

---

# Future Improvements

- FAISS-based Approximate Nearest Neighbor Search
- EfficientNet Feature Extractor
- CLIP Embeddings
- Multi-modal Recommendations
- Larger Fashion Dataset

---

# Author

**Pratik **

B.Tech CSE

AI/ML & Data Science

SKIT Jaipur