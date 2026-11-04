
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

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f3f8f2 0%, #ffffff 45%);
    color: #1f2937;
}

[data-testid="stHeader"] {
    background: rgba(243, 248, 242, 0.9);
}

h1, h2, h3, p, label,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stAlert"] p {
    color: #1f2937 !important;
}

[data-testid="stAlert"] {
    color: #1f2937 !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #1f2937 !important;
}

.leafvision-subtitle {
    color: #52705a !important;
    margin-bottom: 1.5rem;
}

[data-testid="stMetric"] {
    background: #f0f7ef;
    border: 1px solid #d5e5d4;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stFileUploader"] {
    background: #f8fbf7;
    border-radius: 12px;
}

.stButton > button {
    width: 100%;
    background-color: #258348;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #176b37;
    color: white;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Model file not found.")
    return load_model(MODEL_PATH, compile=False)


@st.cache_data
def get_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError("metadata.json not found.")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    for key in ("disease_classes", "species_classes"):
        if not metadata.get(key):
            raise ValueError(f"Missing {key} in metadata.json.")

    return metadata


def pretty_label(label):
    return str(label).replace("_", " ").strip()


def get_named_predictions(model, image_batch):
    predictions = model(image_batch, training=False)

    if isinstance(predictions, dict):
        named_predictions = predictions
    else:
        output_names = getattr(model, "output_names", [])

        if len(output_names) != len(predictions):
            raise ValueError("Could not match model outputs with their names.")

        named_predictions = dict(zip(output_names, predictions))

    required = ("disease_output", "species_output")

    for name in required:
        if name not in named_predictions:
            raise ValueError(f"Missing model output: {name}")

    return named_predictions


st.title("🌿 LeafVision-AI")

st.markdown(
    '<p class="leafvision-subtitle">'
    'Identify plant species and leaf disease from an image.'
    '</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Invalid image. Please upload a valid JPG or PNG.")
        st.stop()

    if st.button("🔍 Analyze Leaf"):

        with st.spinner("Loading AI model and analyzing your leaf..."):

            try:
                model = get_model()
                metadata = get_metadata()

                resized_image = image.resize(
                    (224, 224),
                    Image.Resampling.LANCZOS
                )

                image_batch = np.expand_dims(
                    np.asarray(resized_image, dtype=np.float32),
                    axis=0
                )

                outputs = get_named_predictions(model, image_batch)

                disease_scores = np.asarray(
                    outputs["disease_output"]
                )[0].reshape(-1)

                species_scores = np.asarray(
                    outputs["species_output"]
                )[0].reshape(-1)

                disease_classes = metadata["disease_classes"]
                species_classes = metadata["species_classes"]

                if len(disease_scores) != len(disease_classes):
                    raise ValueError(
                        "Disease output does not match metadata."
                    )

                if len(species_scores) != len(species_classes):
                    raise ValueError(
                        "Species output does not match metadata."
                    )

                disease_index = int(np.argmax(disease_scores))
                species_index = int(np.argmax(species_scores))

                disease_name = disease_classes[disease_index]
                species_name = species_classes[species_index]

                disease_name = disease_name.split("___", 1)[-1]

                disease_confidence = float(
                    disease_scores[disease_index]
                ) * 100

                species_confidence = float(
                    species_scores[species_index]
                ) * 100

                healthy = disease_name.lower().strip("_ ") == "healthy"

                st.divider()
                st.header("🌱 Prediction Results")

                st.metric(
                    "Plant Species",
                    pretty_label(species_name),
                    f"{species_confidence:.1f}% confidence"
                )

                st.metric(
                    "Disease Condition",
                    pretty_label(disease_name),
                    f"{disease_confidence:.1f}% confidence"
                )

                if healthy:
                    st.success("Leaf Status: Healthy 🌿")
                else:
                    st.warning("Leaf Status: Disease Detected ⚠️")

            except Exception as error:
                st.error(f"Prediction failed: {error}")
