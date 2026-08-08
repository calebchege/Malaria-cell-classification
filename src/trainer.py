
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
from .metrics import MetricTracker


#the training engine
class MalariaTrainer:
    """
    Train and validate a PyTorch model.
    """

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler=None, device="cpu", tracker=None,
                 epochs=20, early_stopping=5,
                 checkpoint_path="best_model.pth"):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        

        self.epochs = epochs

        self.early_stopping = early_stopping

        self.checkpoint_path = checkpoint_path

        self.best_val_loss = float("inf")

        self.early_stop_counter = 0

        self.tracker = MetricTracker()

        self.model.to(self.device)


    def train_one_epoch(self):
      """
      Train the model for one epoch.

      Returns
      -------
      dict
          Training metrics.
      """

      self.model.train()

      running_loss = 0.0

      all_predictions = []
      all_labels = []

      progress_bar = tqdm(
                      self.train_loader,
                      desc="Training",
                      leave=False,
                    )

      for images, labels, _ in progress_bar:
          images = images.to(self.device)
          labels = labels.to(self.device)

          self.optimizer.zero_grad()

          outputs = self.model(images)

          loss = self.criterion(outputs, labels)

          loss.backward()

          self.optimizer.step()

          running_loss += loss.item()

          predictions = outputs.argmax(dim=1)

          all_predictions.extend(
              predictions.cpu().numpy()
          )

          all_labels.extend(
              labels.cpu().numpy()
          )
          progress_bar.set_postfix({
          "Loss": f"{loss.item():.4f}"
        })

      epoch_loss = running_loss / len(self.train_loader)

      accuracy = accuracy_score(
          all_labels,
          all_predictions
      )

      precision = precision_score(
          all_labels,
          all_predictions,
          zero_division=0
      )

      recall = recall_score(
          all_labels,
          all_predictions,
          zero_division=0
      )

      f1 = f1_score(
          all_labels,
          all_predictions,
          zero_division=0
      )

      return {
          "loss": epoch_loss,
          "accuracy": accuracy,
          "precision": precision,
          "recall": recall,
          "f1": f1,
      }


    def validate(self):
      """
      Evaluate the model on the validation dataset.

      Returns
      -------
      dict
          Validation metrics.
      """

      self.model.eval()

      running_loss = 0.0

      all_predictions = []
      all_labels = []

      correct = 0
      total = 0

      with torch.no_grad():

          progress_bar = tqdm(
              self.val_loader,
              desc="Validation",
              leave=False,
          )

          for images, labels, _ in progress_bar:

              images = images.to(self.device)
              labels = labels.to(self.device)

              outputs = self.model(images)

              loss = self.criterion(outputs, labels)

              running_loss += loss.item()

              predictions = outputs.argmax(dim=1)

              all_predictions.extend(
                  predictions.cpu().numpy()
              )

              all_labels.extend(
                  labels.cpu().numpy()
              )

              correct += (predictions == labels).sum().item()
              total += labels.size(0)

              running_accuracy = correct / total
              avg_loss = running_loss / (progress_bar.n + 1)

              progress_bar.set_postfix({
                  "Loss": f"{avg_loss:.4f}",
                  "Acc": f"{running_accuracy:.3f}",
              })

      epoch_loss = running_loss / len(self.val_loader)

      accuracy = accuracy_score(
          all_labels,
          all_predictions
      )

      precision = precision_score(
          all_labels,
          all_predictions,
          zero_division=0
      )

      recall = recall_score(
          all_labels,
          all_predictions,
          zero_division=0
      )

      f1 = f1_score(
          all_labels,
          all_predictions,
          zero_division=0
      )

      return {
          "loss": epoch_loss,
          "accuracy": accuracy,
          "precision": precision,
          "recall": recall,
          "f1": f1,
      }


    def save_checkpoint(self):
      """
      Save the current training checkpoint.

      The checkpoint contains the model weights, optimizer state,
      scheduler state, best validation loss, current epoch,
      and training history.
      """

      checkpoint = {
                    "epoch": self.current_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict":  self.scheduler.state_dict()
                                              if self.scheduler is not None
                                              else None,
                                            
                      "best_val_loss": self.best_val_loss,
                      "history": self.tracker.history,
                    }
      torch.save(
          checkpoint,
          self.checkpoint_path
      )

    def fit(self,resume=False):
      """
      Train the model for the configured number of epochs.
      """
      start_epoch = 0

      if resume:
        start_epoch = self.load_checkpoint()

      for epoch in range(start_epoch,self.epochs):
          self.current_epoch = epoch

          print(f"\nEpoch {epoch + 1}/{self.epochs}")
          print("-" * 60)

          # -----------------------------
          # Training
          # -----------------------------

          train_metrics = self.train_one_epoch()

          # -----------------------------
          # Validation
          # -----------------------------

          val_metrics = self.validate()

          # -----------------------------
          # Scheduler
          # -----------------------------

          if self.scheduler is not None:
              self.scheduler.step(val_metrics["loss"])
          # -----------------------------
          # Store Metrics
          # -----------------------------

          self.tracker.update(
              train_loss=train_metrics["loss"],
              val_loss=val_metrics["loss"],

              train_accuracy=train_metrics["accuracy"],
              val_accuracy=val_metrics["accuracy"],

              train_precision=train_metrics["precision"],
              val_precision=val_metrics["precision"],

              train_recall=train_metrics["recall"],
              val_recall=val_metrics["recall"],

              train_f1=train_metrics["f1"],
              val_f1=val_metrics["f1"],
          )

          # -----------------------------
          # Save Best Model
          # -----------------------------

          if val_metrics["loss"] < self.best_val_loss:

              self.best_val_loss = val_metrics["loss"]

              self.early_stop_counter = 0

              self.save_checkpoint()

              print("✓ Best model saved.")

          else:

              self.early_stop_counter += 1

          # -----------------------------
          # Epoch Summary
          # -----------------------------

          print(
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}"
          )

          print(
              f"Train Accuracy: {train_metrics['accuracy']:.4f} | "
              f"Val Accuracy: {val_metrics['accuracy']:.4f}"
          )

          print(
              f"Train F1: {train_metrics['f1']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}"
          )

          # -----------------------------
          # Early Stopping
          # -----------------------------

          if self.early_stop_counter >= self.early_stopping:

              print("\nEarly stopping triggered.")

              break

    def load_checkpoint(self):
      """
      Load a previously saved training checkpoint.

      Returns
      -------
      int
          The last completed epoch.
      """

      import os

      if not os.path.exists(self.checkpoint_path):
          raise FileNotFoundError(
              f"Checkpoint not found: {self.checkpoint_path}"
          )

      checkpoint = torch.load(
          self.checkpoint_path,
          map_location=self.device,
      )

      self.model.load_state_dict(
          checkpoint["model_state_dict"]
      )

      self.optimizer.load_state_dict(
          checkpoint["optimizer_state_dict"]
      )

      if (self.scheduler is not None and checkpoint["scheduler_state_dict"] is not None):
            self.scheduler.load_state_dict(
              checkpoint["scheduler_state_dict"]
                   )

      self.current_epoch = checkpoint["epoch"]

      self.best_val_loss = checkpoint["best_val_loss"]

      self.tracker.history = checkpoint["history"]

      return self.current_epoch +1

    def plot_history(self):
      """
      Plot the training history for all tracked metrics.
      """

      history = self.tracker.history

      epochs = range(
          1,
          len(history["train_loss"]) + 1
      )

      fig, axes = plt.subplots(
          3,
          2,
          figsize=(15, 15)
      )

      # -----------------------------
      # Loss
      # -----------------------------
      axes[0, 0].plot(
          epochs,
          history["train_loss"],
          label="Train"
      )

      axes[0, 0].plot(
          epochs,
          history["val_loss"],
          label="Validation"
      )

      axes[0, 0].set_title("Loss")
      axes[0, 0].set_xlabel("Epoch")
      axes[0, 0].set_ylabel("Loss")
      axes[0, 0].legend()
      axes[0, 0].grid(True)

      # -----------------------------
      # Accuracy
      # -----------------------------
      axes[0, 1].plot(
          epochs,
          history["train_accuracy"],
          label="Train"
      )

      axes[0, 1].plot(
          epochs,
          history["val_accuracy"],
          label="Validation"
      )

      axes[0, 1].set_title("Accuracy")
      axes[0, 1].set_xlabel("Epoch")
      axes[0, 1].set_ylabel("Accuracy")
      axes[0, 1].legend()
      axes[0, 1].grid(True)

      # -----------------------------
      # Precision
      # -----------------------------
      axes[1, 0].plot(
          epochs,
          history["train_precision"],
          label="Train"
      )

      axes[1, 0].plot(
          epochs,
          history["val_precision"],
          label="Validation"
      )

      axes[1, 0].set_title("Precision")
      axes[1, 0].set_xlabel("Epoch")
      axes[1, 0].set_ylabel("Precision")
      axes[1, 0].legend()
      axes[1, 0].grid(True)

      # -----------------------------
      # Recall
      # -----------------------------
      axes[1, 1].plot(
          epochs,
          history["train_recall"],
          label="Train"
      )

      axes[1, 1].plot(
          epochs,
          history["val_recall"],
          label="Validation"
      )

      axes[1, 1].set_title("Recall")
      axes[1, 1].set_xlabel("Epoch")
      axes[1, 1].set_ylabel("Recall")
      axes[1, 1].legend()
      axes[1, 1].grid(True)

      # -----------------------------
      # F1 Score
      # -----------------------------
      axes[2, 0].plot(
          epochs,
          history["train_f1"],
          label="Train"
      )

      axes[2, 0].plot(
          epochs,
          history["val_f1"],
          label="Validation"
      )

      axes[2, 0].set_title("F1 Score")
      axes[2, 0].set_xlabel("Epoch")
      axes[2, 0].set_ylabel("F1 Score")
      axes[2, 0].legend()
      axes[2, 0].grid(True)

      # -----------------------------
      # Empty subplot (reserved)
      # -----------------------------
      axes[2, 1].axis("off")
      axes[2, 1].text(
          0.5,
          0.5,
          "Reserved for Future Metrics\n(e.g. ROC-AUC,\nLearning Rate, MCC)",
          ha="center",
          va="center",
          fontsize=12
      )

      plt.tight_layout()
      plt.show()