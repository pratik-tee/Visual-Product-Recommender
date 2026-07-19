# ==========================================
# feature_extractor.py
# ==========================================

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model

from config import TRANSFER_MODEL, SIAMESE_FINAL_MODEL


def load_feature_extractor(model_type="finetuned"):
    """
    Load feature extractor.

    Parameters
    ----------
    model_type : str
        "baseline"  -> ImageNet ResNet50
        "finetuned" -> Fine-tuned ResNet50 (transfer learning stage)
        "siamese"   -> Siamese-refined embedding model

    Returns
    -------
    keras.Model
    """

    if model_type == "baseline":

        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        output = GlobalAveragePooling2D()(base_model.output)

        model = Model(
            inputs=base_model.input,
            outputs=output
        )

        return model

    elif model_type == "finetuned":

        classifier = load_model(TRANSFER_MODEL)

        model = Model(
            inputs=classifier.input,
            outputs=classifier.layers[-3].output
        )

        return model

    elif model_type == "siamese":


        model = load_model(SIAMESE_FINAL_MODEL)

        return model

    else:

        raise ValueError(
            "model_type must be 'baseline', 'finetuned', or 'siamese'"
        )


def extract_embedding(model, image):

    embedding = model.predict(
        image,
        verbose=0
    )

    return embedding.flatten()