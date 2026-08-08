from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM implementation for the MalariaCNN model.

    Grad-CAM identifies image regions that contributed most
    strongly to a selected class prediction.
    """

    def __init__(
        self,
        model,
        device="cpu",
        target_layer=None,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained classification model.

        device : torch.device or str
            Device used for inference.

        target_layer : torch.nn.Module, optional
            Convolutional layer used for Grad-CAM.

            If None, the last Conv2d layer in the model
            is automatically selected.
        """

        self.model = model
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.target_layer = (
            target_layer
            if target_layer is not None
            else self._find_last_conv_layer()
        )

        self.activations = None
        self.gradients = None

        self._register_hooks()

    # --------------------------------------------------
    # Find target layer
    # --------------------------------------------------

    def _find_last_conv_layer(self):
        """
        Find the last Conv2d layer in the model.

        Returns
        -------
        torch.nn.Module
            Last convolutional layer.
        """

        last_conv = None

        for module in self.model.modules():

            if isinstance(
                module,
                torch.nn.Conv2d
            ):
                last_conv = module

        if last_conv is None:
            raise ValueError(
                "No Conv2d layer found in the model."
            )

        return last_conv

    # --------------------------------------------------
    # Register hooks
    # --------------------------------------------------

    def _register_hooks(self):
        """
        Register forward and backward hooks on the
        target convolutional layer.
        """

        def forward_hook(
            module,
            input,
            output,
        ):
            self.activations = output

        def backward_hook(
            module,
            grad_input,
            grad_output,
        ):
            self.gradients = grad_output[0]

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                backward_hook
            )
        )

    # --------------------------------------------------
    # Generate CAM
    # --------------------------------------------------

    def generate(
        self,
        image_tensor,
        target_class=None,
    ):
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Preprocessed image tensor with shape:

            (1, 3, H, W)

        target_class : int, optional
            Class whose activation should be visualized.

            If None, the model's predicted class is used.

        Returns
        -------
        heatmap : numpy.ndarray
            Normalized heatmap with values between 0 and 1.

        predicted_class : int
            Model prediction.

        probabilities : numpy.ndarray
            Softmax probabilities.
        """

        if image_tensor.ndim != 4:
            raise ValueError(
                "image_tensor must have shape "
                "(1, 3, H, W)."
            )

        image_tensor = image_tensor.to(
            self.device
        )

        # Make sure gradients are enabled
        image_tensor = image_tensor.clone()
        image_tensor.requires_grad_(True)

        # Clear previous gradients
        self.model.zero_grad()

        # Forward pass
        logits = self.model(
            image_tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        predicted_class = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        # Use predicted class if no target is supplied
        if target_class is None:
            target_class = predicted_class

        if target_class < 0 or target_class >= logits.shape[1]:
            raise ValueError(
                f"Invalid target class: {target_class}"
            )

        # Select score for target class
        target_score = logits[
            0,
            target_class
        ]

        # Backward pass
        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        # --------------------------------------------------
        # Global average pooling of gradients
        # --------------------------------------------------

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # --------------------------------------------------
        # Weighted combination of feature maps
        # --------------------------------------------------

        cam = (
            weights * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Keep only positive influence
        cam = F.relu(cam)

        # --------------------------------------------------
        # Resize CAM to input image dimensions
        # --------------------------------------------------

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Remove unnecessary dimensions
        heatmap = cam[
            0,
            0
        ].detach().cpu().numpy()

        # --------------------------------------------------
        # Normalize to [0, 1]
        # --------------------------------------------------

        heatmap -= heatmap.min()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return (
            heatmap,
            predicted_class,
            probabilities[
                0
            ].detach().cpu().numpy(),
        )

    # --------------------------------------------------
    # Overlay heatmap
    # --------------------------------------------------

    def overlay(
        self,
        image,
        heatmap,
        alpha=0.45,
    ):
        """
        Overlay a Grad-CAM heatmap on an image.

        Parameters
        ----------
        image : PIL.Image.Image or str or Path
            Original image.

        heatmap : numpy.ndarray
            Grad-CAM heatmap.

        alpha : float
            Heatmap transparency.

        Returns
        -------
        numpy.ndarray
            RGB visualization.
        """

        if isinstance(
            image,
            (str, Path)
        ):
            image = Image.open(image)

        image = image.convert("RGB")

        image_array = np.asarray(
            image
        ).astype(
            np.float32
        ) / 255.0

        # Convert heatmap to PIL image
        heatmap_uint8 = (
            heatmap * 255
        ).astype(
            np.uint8
        )

        heatmap_image = Image.fromarray(
            heatmap_uint8
        )

        heatmap_image = (
            heatmap_image.resize(
                image.size,
                Image.Resampling.BILINEAR,
            )
        )

        heatmap_array = (
            np.asarray(
                heatmap_image
            ).astype(
                np.float32
            ) / 255.0
        )

        # Apply matplotlib colormap
        cmap = plt.get_cmap(
            "jet"
        )

        colored_heatmap = cmap(
            heatmap_array
        )[:, :, :3]

        # Blend
        overlay = (
            (1 - alpha)
            * image_array
            +
            alpha
            * colored_heatmap
        )

        overlay = np.clip(
            overlay,
            0,
            1,
        )

        return overlay