# ==========================================
# app.py
# ==========================================

import os
import tempfile

import streamlit as st
from PIL import Image

from recommendation import recommend_products

st.set_page_config(
    page_title="Visual Product Recommender",
    layout="wide"
)

st.title("🛍️ Visual Product Recommendation System")
st.write("Upload a fashion product image to find similar products.")

# ==========================================
# Model Selector
# ==========================================

model_type = st.radio(
    "Select embedding model",
    options=["baseline", "finetuned", "siamese"],
    index=1,          # defaults to "finetuned"
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    recommendations = recommend_products(
        temp_path,
        model_type=model_type
    )

    os.remove(temp_path)

    st.subheader(f"Recommended Products ({model_type})")

    cols = st.columns(5)

    for i, product in enumerate(recommendations):
        with cols[i]:
            st.image(
                product["image_path"],
                use_container_width=True
            )
            st.write(f"**ID:** {product['id']}")
            st.write(product["articleType"])
            st.write(product["gender"])
            st.write(f"Similarity: {product['score']:.3f}")