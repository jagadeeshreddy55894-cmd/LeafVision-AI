import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "LeafVision_ResNet50.keras"
METADATA_PATH = PROJECT_DIR / "metadata.json"


st.set_page_config(
    page_title="LeafVision-AI",
    page_icon="🌿",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f3f8f2 0%, #ffffff 36%);
    }
    [data-testid="stHeader"] {
        background: rgba(243, 248, 242, 0.8);
    }
    .leafvision-subtitle {
        color: #52705a;
        margin-top: -0.75rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: #f3f8f2;
        border: 1px solid #dce9dc;
        border-radius: 0.75rem;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return load_model(MODEL_PATH, compile=False)


@st.cache_data
def get_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH.name}")
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    required_keys = ("disease_classes", "species_classes")
    missing_keys = [key for key in required_keys if not metadata.get(key)]
    if missing_keys:
        raise ValueError(
            "metadata.json is missing required class lists: "
            + ", ".join(missing_keys)
        )
    return metadata


def pretty_label(label):
    return label.replace("_", " ").strip()


def get_named_predictions(model, image_batch):
    predictions = model.predict(image_batch, verbose=0)
    if isinstance(predictions, dict):
        named_predictions = predictions
    else:
        output_names = getattr(model, "output_names", [])
        if len(output_names) != len(predictions):
            raise ValueError("The model's outputs could not be matched to their names.")
        named_predictions = dict(zip(output_names, predictions))

    expected_outputs = ("disease_output", "species_output")
    missing_outputs = [name for name in expected_outputs if name not in named_predictions]
    if missing_outputs:
        raise ValueError(
            "The model is missing expected output(s): " + ", ".join(missing_outputs)
        )
    return named_predictions


st.title("🌿 LeafVision-AI")
st.markdown(
    '<p class="leafvision-subtitle">Identify plant species and leaf disease from an image.</p>',
    unsafe_allow_html=True,
)

try:
    model = get_model()
    metadata = get_metadata()
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Could not load the model or its metadata: {error}")
    st.stop()
except Exception as error:
    st.error(f"An unexpected error occurred while loading the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, and PNG.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        resized_image = image.resize((224, 224), Image.Resampling.LANCZOS)
        image_batch = np.expand_dims(np.asarray(resized_image, dtype=np.float32), axis=0)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Could not read the uploaded image. Please try a valid JPG or PNG: {error}")
        st.stop()

    st.image(image, caption="Uploaded leaf image", use_container_width=True)

    try:
        outputs = get_named_predictions(model, image_batch)
        disease_scores = np.asarray(outputs["disease_output"])[0].reshape(-1)
        species_scores = np.asarray(outputs["species_output"])[0].reshape(-1)
        disease_classes = metadata["disease_classes"]
        species_classes = metadata["species_classes"]

        if len(disease_scores) != len(disease_classes):
            raise ValueError(
                "The disease output size does not match disease_classes in metadata.json."
            )
        if len(species_scores) != len(species_classes):
            raise ValueError(
                "The species output size does not match species_classes in metadata.json."
            )

        disease_index = int(np.argmax(disease_scores))
        species_index = int(np.argmax(species_scores))
        disease_class = disease_classes[disease_index]
        species_class = species_classes[species_index]
        disease_name = disease_class.split("___", 1)[-1]
        healthy = disease_name.lower().strip("_ ") == "healthy"
        status = "Healthy" if healthy else "Diseased"

        st.subheader("Prediction")
        species_column, disease_column = st.columns(2)
        species_column.metric(
            "Plant species",
            pretty_label(species_class),
            f"{species_scores[species_index] * 100:.1f}% confidence",
        )
        disease_column.metric(
            "Disease condition",
            pretty_label(disease_name),
            f"{disease_scores[disease_index] * 100:.1f}% confidence",
        )
        if healthy:
            st.success(f"Leaf status: {status}")
        else:
            st.warning(f"Leaf status: {status}")
    except (IndexError, KeyError, TypeError, ValueError) as error:
        st.error(f"Could not generate a valid prediction: {error}")
    except Exception as error:
        st.error(f"Prediction failed unexpectedly: {error}")
