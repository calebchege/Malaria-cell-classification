"""
Utility functions for the Malaria Cell Classification project.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Random seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Return the available PyTorch device.

    Returns
    -------
    torch.device
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(f"Using GPU : {torch.cuda.get_device_name(0)}")

    else:

        device = torch.device("cpu")

        print("Using CPU")

    return device


def count_parameters(model):
    """
    Count trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def save_json(data, filepath):
    """
    Save a dictionary as JSON.

    Parameters
    ----------
    data : dict

    filepath : str or Path
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(filepath, "w") as f:

        json.dump(
            data,
            f,
            indent=4,
        )


def load_json(filepath):
    """
    Load a JSON file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    dict
    """

    filepath = Path(filepath)

    with open(filepath, "r") as f:

        return json.load(f)


def format_time(seconds):
    """
    Convert seconds to HH:MM:SS.

    Parameters
    ----------
    seconds : float

    Returns
    -------
    str
    """

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    seconds = int(seconds % 60)

    return f"{hours:02}:{minutes:02}:{seconds:02}"