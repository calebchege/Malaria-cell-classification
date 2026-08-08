import torch
import torch.nn.functional as F

from src.model import MalariaCNN

from .config import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DEVICE,
    MODEL_CONFIG,
)


class MalariaPredictor:
    """
    Load the trained malaria CNN and perform inference.
    """

    def __init__(
        self,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    ):
        """
        Parameters
        ----------
        checkpoint_path : Path or str
            Path to the trained model checkpoint.

        device : torch.device or str
            Device used for inference.
        """

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

        self.class_names = CLASS_NAMES

        # ---------------------------------------------
        # Build model
        # ---------------------------------------------

        self.model = MalariaCNN(
            **MODEL_CONFIG
        )

        self.model.to(self.device)

        # ---------------------------------------------
        # Load checkpoint
        # ---------------------------------------------

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        # ---------------------------------------------
        # Load model weights
        # ---------------------------------------------

        if "model_state_dict" in checkpoint:

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            # Support a raw state_dict checkpoint
            self.model.load_state_dict(
                checkpoint
            )

        # ---------------------------------------------
        # Evaluation mode
        # ---------------------------------------------

        self.model.eval()

    @torch.no_grad()
    def predict(self, image_tensor):
        """
        Generate a prediction for one image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Preprocessed image tensor with shape:
            (1, 3, H, W)

        Returns
        -------
        dict
            Prediction results containing:

            prediction
            predicted_class
            confidence
            probabilities
        """

        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(
                "image_tensor must be a torch.Tensor."
            )

        if image_tensor.ndim != 4:
            raise ValueError(
                "image_tensor must have shape "
                "(batch_size, channels, height, width)."
            )

        # ---------------------------------------------
        # Move image to inference device
        # ---------------------------------------------

        image_tensor = image_tensor.to(
            self.device
        )

        # ---------------------------------------------
        # Forward pass
        # ---------------------------------------------

        logits = self.model(image_tensor)

        # ---------------------------------------------
        # Convert logits to probabilities
        # ---------------------------------------------

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        # ---------------------------------------------
        # Determine predicted class
        # ---------------------------------------------

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

        predicted_index = int(
            prediction.item()
        )

        confidence_value = float(
            confidence.item()
        )

        probability_values = (
            probabilities[0]
            .detach()
            .cpu()
            .tolist()
        )

        return {
            "prediction": predicted_index,
            "predicted_class": self.class_names[
                predicted_index
            ],
            "confidence": confidence_value,
            "probabilities": {
                self.class_names[0]:
                    probability_values[0],

                self.class_names[1]:
                    probability_values[1],
            },
        }