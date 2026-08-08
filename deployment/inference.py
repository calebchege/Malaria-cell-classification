from pathlib import Path

from PIL import Image

from .config import (
    ARTIFACTS_DIR,
    DEVICE,
)

from .preprocessing import (
    InferencePreprocessor,
)

from .predictor import (
    MalariaPredictor,
)

from .gradcam import (
    GradCAM,
)


class MalariaInferenceService:
    """
    Unified inference service for the malaria cell
    classification deployment.

    Combines:

    1. Image preprocessing
    2. Model prediction
    3. Grad-CAM explanation
    """

    def __init__(
        self,
        artifacts_dir=ARTIFACTS_DIR,
        device=DEVICE,
    ):
        """
        Initialize all deployment components.
        """

        self.device = device

        # --------------------------------------------------
        # Preprocessor
        # --------------------------------------------------

        self.preprocessor = (
            InferencePreprocessor(
                artifacts_dir=artifacts_dir,
            )
        )

        # --------------------------------------------------
        # Predictor
        # --------------------------------------------------

        self.predictor = (
            MalariaPredictor(
                device=device,
            )
        )

        # --------------------------------------------------
        # Grad-CAM
        # --------------------------------------------------

        self.gradcam = GradCAM(
            model=self.predictor.model,
            device=device,
        )

    def predict(
        self,
        image,
        generate_gradcam=True,
        gradcam_alpha=0.45,
    ):
        """
        Run the complete inference pipeline.

        Parameters
        ----------
        image : PIL.Image.Image, str, or Path
            Input image.

        generate_gradcam : bool
            Whether to generate a Grad-CAM explanation.

        gradcam_alpha : float
            Transparency of Grad-CAM overlay.

        Returns
        -------
        dict
            Complete inference results.
        """

        # --------------------------------------------------
        # Load original image
        # --------------------------------------------------

        original_image = (
            self.preprocessor.load_image(
                image
            )
        )

        # --------------------------------------------------
        # Preprocess
        # --------------------------------------------------

        image_tensor = (
            self.preprocessor.preprocess(
                original_image
            )
        )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = (
            self.predictor.predict(
                image_tensor
            )
        )

        result = {
            "prediction": prediction,
            "original_image": original_image,
            "image_tensor": image_tensor,
        }

        # --------------------------------------------------
        # Grad-CAM
        # --------------------------------------------------

        if generate_gradcam:

            heatmap, predicted_class, probabilities = (
                self.gradcam.generate(
                    image_tensor,
                    target_class=prediction[
                        "prediction"
                    ],
                )
            )

            overlay = self.gradcam.overlay(
                image=original_image,
                heatmap=heatmap,
                alpha=gradcam_alpha,
            )

            result["gradcam"] = {
                "heatmap": heatmap,
                "overlay": overlay,
                "target_class": predicted_class,
                "probabilities": probabilities,
            }

        else:

            result["gradcam"] = None

        return result