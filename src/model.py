
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from tqdm.auto import tqdm

class MalariaCNN(nn.Module):
    """
    Configurable CNN for malaria cell classification.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=2,
        conv_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.30,
        activation="relu",
        use_batchnorm=True,
    ):

        super().__init__()

        if len(conv_channels) == 0:
            raise ValueError(
                "conv_channels must contain at least one value."
            )

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size should be odd."
            )

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.use_batchnorm = use_batchnorm

        self.features = self._build_conv_layers()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(
                self.conv_channels[-1],
                self.num_classes
            )
        )


    def _get_activation(self):
      """
      Return the activation function specified by self.activation.

      Returns
      -------
      nn.Module
          PyTorch activation module.
      """

      activations = {
          "relu": nn.ReLU(inplace=True),
          "leaky_relu": nn.LeakyReLU(
              negative_slope=0.01,
              inplace=True
          ),
          "gelu": nn.GELU(),
      }

      activation = self.activation.lower()

      if activation not in activations:
          raise ValueError(
              f"Unsupported activation '{self.activation}'. "
              f"Choose from {list(activations.keys())}."
          )

      return activations[activation]

    def _build_conv_layers(self):
      """
      Build the convolutional feature extractor.

      Returns
      -------
      nn.Sequential
          Feature extraction network.
      """

      layers = []

      in_channels = self.input_channels

      for out_channels in self.conv_channels:

          # Convolution
          layers.append(
              nn.Conv2d(
                  in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=self.kernel_size,
                  padding=self.kernel_size // 2,
                  bias=not self.use_batchnorm,
              )
          )

          # Batch Normalization
          if self.use_batchnorm:
              layers.append(
                  nn.BatchNorm2d(out_channels)
              )

          # Activation
          layers.append(
              self._get_activation()
          )

          # Pooling
          layers.append(
              nn.MaxPool2d(
                  kernel_size=2,
                  stride=2
              )
          )

          # Dropout
          if self.dropout > 0:
              layers.append(
                  nn.Dropout2d(
                      p=self.dropout
                  )
              )

          # Prepare for the next block
          in_channels = out_channels

      return nn.Sequential(*layers)

    def forward(self, x):
      """
      Forward pass.

      Parameters
      ----------
      x : torch.Tensor
          Input tensor of shape
          (batch_size, channels, height, width)

      Returns
      -------
      torch.Tensor
          Logits for each class.
      """

      # Feature extraction
      x = self.features(x)

      # Global average pooling
      x = self.global_pool(x)

      # Classification
      x = self.classifier(x)

      return x