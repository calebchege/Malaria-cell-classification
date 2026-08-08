

import random
from pathlib import Path

import numpy as np
import pandas as pd


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split

class MalariaDataset(Dataset):
    """
    PyTorch Dataset for malaria cell images.

    Each sample consists of:
        image tensor
        label
    """

    def __init__(self, dataframe, transform=None):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing image metadata.

        transform : torchvision.transforms.Compose, optional
            Transformations applied to each image.
        """

        self.data = dataframe
        self.transform = transform

    def __len__(self):
      """
      Return the number of samples.
      """

      return len(self.data)


    def __getitem__(self, index):
      """
      Retrieve a single sample.

      Parameters
      ----------
      index : int

      Returns
      -------
      image : torch.Tensor
      label : int
      image_id : int
      """

      sample = self.data.iloc[index]

      with Image.open(sample["filepath"]) as img:
          image = img.convert("RGB")

      if self.transform is not None:
          image = self.transform(image)

      label = int(sample["label"])
      image_id = int(sample["image_id"])

      return image, label, image_id