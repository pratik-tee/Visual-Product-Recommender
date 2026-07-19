# ==========================================
# pair_generator.py
# ==========================================

import os
import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence

from config import *
from preprocess import preprocess_image


class PairGenerator(Sequence):

    def __init__(self, csv_file, batch_size=BATCH_SIZE, **kwargs):
        super().__init__(**kwargs)

        self.df = pd.read_csv(csv_file)
        self.batch_size = batch_size

    def __len__(self):

        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):

        batch = self.df.iloc[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        img1 = []
        img2 = []
        labels = []

        for _, row in batch.iterrows():

            path1 = os.path.join(
                IMAGE_FOLDER,
                f"{int(row['image1'])}.jpg"
            )

            path2 = os.path.join(
                IMAGE_FOLDER,
                f"{int(row['image2'])}.jpg"
            )

            image1 = preprocess_image(path1)[0]
            image2 = preprocess_image(path2)[0]

            img1.append(image1)
            img2.append(image2)
            labels.append(row["label"])

        return (
            (
                np.array(img1, dtype=np.float32),
                np.array(img2, dtype=np.float32),
            ),
            np.array(labels, dtype=np.float32),
        )