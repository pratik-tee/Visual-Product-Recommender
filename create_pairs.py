# ==========================================
# create_pairs.py
# ==========================================

import random
import pandas as pd

from config import SUBSET_CSV, PAIR_CSV, RANDOM_STATE

random.seed(RANDOM_STATE)

# Load subset
df = pd.read_csv(SUBSET_CSV)

pairs = []

# Group by article type
groups = df.groupby("articleType")

# ---------- Positive Pairs ----------
for _, group in groups:

    ids = group["id"].tolist()

    if len(ids) < 2:
        continue

    for _ in range(min(len(ids), 10)):

        img1, img2 = random.sample(ids, 2)

        pairs.append([
            img1,
            img2,
            1
        ])

# ---------- Negative Pairs ----------
ids = df["id"].tolist()

while len(pairs) < 2 * len(df):

    img1 = random.choice(ids)
    img2 = random.choice(ids)

    type1 = df.loc[df["id"] == img1, "articleType"].values[0]
    type2 = df.loc[df["id"] == img2, "articleType"].values[0]

    if type1 != type2:

        pairs.append([
            img1,
            img2,
            0
        ])

pairs = pd.DataFrame(
    pairs,
    columns=[
        "image1",
        "image2",
        "label"
    ]
)

pairs.to_csv(
    PAIR_CSV,
    index=False
)

print(f"Pairs Created : {len(pairs)}")
print(f"Saved to : {PAIR_CSV}")