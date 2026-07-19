# ==========================================
# train_siamese.py  
# ==========================================

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from config import *
from pair_generator import PairGenerator

# ==========================================
# Load Fine-tuned Backbone
# ==========================================


classifier = load_model(TRANSFER_MODEL)

embedding_model = Model(
    inputs=classifier.input,
    outputs=classifier.layers[-3].output   
)



UNFREEZE_LAST_N = 15

for layer in embedding_model.layers[:-UNFREEZE_LAST_N]:
    layer.trainable = False

for layer in embedding_model.layers[-UNFREEZE_LAST_N:]:
    layer.trainable = True

# ==========================================
# Siamese Inputs
# ==========================================

input_a = Input(shape=(224, 224, 3))
input_b = Input(shape=(224, 224, 3))

embedding_a = embedding_model(input_a)
embedding_b = embedding_model(input_b)

distance = Lambda(
    lambda tensors: tf.abs(tensors[0] - tensors[1])
)([embedding_a, embedding_b])

output = Dense(1, activation="sigmoid")(distance)

model = Model(
    inputs=[input_a, input_b],
    outputs=output
)

# ==========================================
# Compile
# ==========================================


model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# Generator
# ==========================================

train_generator = PairGenerator(PAIR_CSV)

# ==========================================
# Train
# ==========================================

history = model.fit(
    train_generator,
    epochs=EPOCHS
)



model.save(SIAMESE_MODEL)
embedding_model.save(SIAMESE_FINAL_MODEL)

print("\nSiamese Model Saved Successfully!")
print("Embedding sub-model saved to:", SIAMESE_FINAL_MODEL)