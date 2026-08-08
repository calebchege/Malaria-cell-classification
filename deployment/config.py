from pathlib import Path

import torch


# =========================================================
# Project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = (
    PROJECT_DIR / "checkpoints"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "best_model.pth"
)

ARTIFACTS_DIR = (
    PROJECT_DIR / "artifacts"
)

NORMALIZATION_PATH = (
    ARTIFACTS_DIR / "normalization.json"
)

TRANSFORMS_PATH = (
    ARTIFACTS_DIR / "transforms.json"
)


# =========================================================
# Model configuration
# =========================================================

MODEL_CONFIG = {
    "input_channels": 3,
    "num_classes": 2,
    "conv_channels": [32, 64, 128],
    "kernel_size": 3,
    "dropout": 0.30,
    "activation": "relu",
    "use_batchnorm": True,
}


# =========================================================
# Image configuration
# =========================================================

IMAGE_SIZE = (128, 128)


# =========================================================
# Class configuration
# =========================================================

CLASS_NAMES = [
    "Uninfected",
    "Parasitized",
]


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Validation
# =========================================================

def validate_deployment_files():
    """
    Validate that all files required for inference exist.

    Returns
    -------
    bool
        True when all required deployment files exist.

    Raises
    ------
    FileNotFoundError
        If a required deployment file is missing.
    """

    required_files = {
        "Model checkpoint": CHECKPOINT_PATH,
        "Normalization artifact": NORMALIZATION_PATH,
        "Transforms artifact": TRANSFORMS_PATH,
    }

    missing_files = {
        name: path
        for name, path in required_files.items()
        if not path.exists()
    }

    if missing_files:

        message = [
            "Missing deployment files:"
        ]

        for name, path in missing_files.items():

            message.append(
                f"- {name}: {path}"
            )

        raise FileNotFoundError(
            "\n".join(message)
        )

    return True