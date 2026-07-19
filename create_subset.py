import os
import random
import pandas as pd

from config import *

random.seed(RANDOM_STATE)


def create_subset():

    print("Reading styles.csv...")

    df = pd.read_csv(STYLE_CSV, on_bad_lines="skip")

    # Remove rows with missing values
    df = df.dropna(subset=["articleType"])

    subset = pd.DataFrame()

    for category in CATEGORIES:

        print(f"Processing {category}")

        temp = df[df["articleType"] == category]

        # Keep only images that actually exist
        temp = temp[
            temp["id"].apply(
                lambda x: os.path.exists(
                    os.path.join(IMAGE_FOLDER, f"{x}.jpg")
                )
            )
        ]

        if len(temp) >= IMAGES_PER_CATEGORY:
            temp = temp.sample(
                IMAGES_PER_CATEGORY,
                random_state=RANDOM_STATE
            )

        subset = pd.concat([subset, temp])

    subset.reset_index(drop=True, inplace=True)

    subset.to_csv(SUBSET_CSV, index=False)

    print("=" * 40)
    print("Subset Created Successfully")
    print(f"Total Images : {len(subset)}")
    print(f"Saved at : {SUBSET_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    create_subset()