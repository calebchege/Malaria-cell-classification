from pathlib import Path
import json

import torch
from PIL import Image
from torchvision import transforms


class InferencePreprocessor:
    """
    Preprocess a single image for model inference.

    The preprocessing configuration is loaded from the
    normalization artifact generated during training.
    """

    def __init__(self, artifacts_dir):
        """
        Parameters
        ----------
        artifacts_dir : Path or str
            Directory containing preprocessing artifacts.
        """

        self.artifacts_dir = Path(artifacts_dir)

        # --------------------------------------------------
        # Load normalization artifact
        # --------------------------------------------------

        normalization_path = (
            self.artifacts_dir / "normalization.json"
        )

        if not normalization_path.exists():
            raise FileNotFoundError(
                "Normalization artifact not found:\n"
                f"{normalization_path}"
            )

        with open(normalization_path, "r") as f:
            normalization = json.load(f)

        # --------------------------------------------------
        # Load preprocessing configuration
        # --------------------------------------------------

        self.image_size = tuple(
            normalization["image_size"]
        )

        self.mean = tuple(
            normalization["mean"]
        )

        self.std = tuple(
            normalization["std"]
        )

        # --------------------------------------------------
        # Build inference transform
        # --------------------------------------------------

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean,
                std=self.std,
            ),
        ])

    def load_image(self, image):
        """
        Load an image and convert it to RGB.

        Parameters
        ----------
        image : PIL.Image.Image, str, or Path

        Returns
        -------
        PIL.Image.Image
            RGB image.
        """

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL Image, file path, "
                "or pathlib.Path."
            )

        return image.convert("RGB")

    def preprocess(self, image):
        """
        Preprocess a single image for model inference.

        Parameters
        ----------
        image : PIL.Image.Image, str, or Path

        Returns
        -------
        torch.Tensor
            Tensor with shape (1, 3, H, W).
        """

        image = self.load_image(image)

        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor