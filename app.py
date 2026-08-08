import streamlit as st
from PIL import Image

from deployment.inference import (
    MalariaInferenceService,
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon="🔬",
    layout="wide",
)


# =========================================================
# Application title
# =========================================================

st.title("🔬 Malaria Cell Classification")

st.write(
    """
    Upload a microscope image of a red blood cell to
    classify it as **Parasitized** or **Uninfected**.
    """
)

st.info(
    """
    This application is a machine-learning demonstration.
    Its prediction and Grad-CAM visualization should not be
    treated as a medical diagnosis.
    """
)


# =========================================================
# Load inference service
# =========================================================

@st.cache_resource
def load_inference_service():

    return MalariaInferenceService()


inference_service = load_inference_service()


# =========================================================
# Image uploader
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a cell image",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)


# =========================================================
# Prediction
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        width=400,
    )

    analyze = st.button(
        "🔍 Analyze Image",
        type="primary",
    )

    if analyze:

        with st.spinner(
            "Analyzing image..."
        ):

            result = (
                inference_service.predict(
                    image,
                    generate_gradcam=True,
                )
            )

        prediction = result[
            "prediction"
        ]

        gradcam = result[
            "gradcam"
        ]

        # =================================================
        # Prediction result
        # =================================================

        st.subheader(
            "Prediction"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Class",
                prediction[
                    "predicted_class"
                ],
            )

        with col2:

            st.metric(
                "Confidence",
                f"{prediction['confidence']:.2%}",
            )

        # =================================================
        # Probabilities
        # =================================================

        st.subheader(
            "Class Probabilities"
        )

        probabilities = (
            prediction[
                "probabilities"
            ]
        )

        for class_name, probability in (
            probabilities.items()
        ):

            st.write(
                f"**{class_name}: "
                f"{probability:.2%}**"
            )

            st.progress(
                probability
            )

        # =================================================
        # Grad-CAM
        # =================================================

        st.subheader(
            "Model Explanation — Grad-CAM"
        )

        st.write(
            """
            The highlighted regions show areas of the
            image that contributed most strongly to the
            model's prediction. They indicate model
            attention and should not be interpreted as
            definitive identification of a parasite.
            """
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                result[
                    "original_image"
                ],
                caption="Original Image",
                width="stretch",
            )

        with col2:

            st.image(
                gradcam[
                    "overlay"
                ],
                caption="Grad-CAM Explanation",
                width="stretch",
            )