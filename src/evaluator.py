from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image


class ModelEvaluator:
    """
    Evaluate a trained malaria classification model on the
    unseen test dataset.

    Responsibilities
    ----------------
    - Run inference
    - Compute evaluation metrics
    - Store predictions
    - Store probabilities
    - Generate evaluation reports and visualizations
    """

    def __init__(
        self,
        model,
        dataloader,
        criterion,
        device="cpu",
        class_names=None,
        output_dir=None,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained CNN model.

        dataloader : torch.utils.data.DataLoader
            Test DataLoader.

        criterion : torch.nn.Module
            Loss function.

        device : str
            "cpu" or "cuda".

        class_names : list[str], optional
            Human-readable class names.

        output_dir : str or Path, optional
            Directory where evaluation outputs are saved.
        """

        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = torch.device(device)

        # --------------------------------------------------
        # Class names
        # --------------------------------------------------

        if class_names is None:
            class_names = [
                "Uninfected",
                "Parasitized",
            ]

        self.class_names = class_names

        # --------------------------------------------------
        # Output directory
        # --------------------------------------------------

        if output_dir is None:
            output_dir = Path("outputs")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Evaluation results
        # --------------------------------------------------

        self.loss = None

        self.metrics = {}

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        self.targets = []

        self.predictions = []

        self.probabilities = []

        self.image_ids = []

        # --------------------------------------------------
        # Cached DataFrame
        # --------------------------------------------------

        self.results_df = None



    def evaluate(self):
        """
        Evaluate the trained model on the test dataset.

        Returns
        -------
        dict
            Dictionary containing the evaluation metrics.
        """

        # --------------------------------------------------
        # Evaluation mode
        # --------------------------------------------------

        self.model.eval()

        running_loss = 0.0

        # Clear previous results
        self.targets.clear()
        self.predictions.clear()
        self.probabilities.clear()
        self.image_ids.clear()

        # --------------------------------------------------
        # Disable gradients
        # --------------------------------------------------

        with torch.no_grad():

            for images, labels, image_ids in tqdm(
                self.dataloader,
                desc="Evaluating",
            ):

                # ------------------------------------------
                # Move to device
                # ------------------------------------------

                images = images.to(self.device)
                labels = labels.to(self.device)

                # ------------------------------------------
                # Forward pass
                # ------------------------------------------

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                # ------------------------------------------
                # Softmax probabilities
                # ------------------------------------------

                probs = F.softmax(outputs, dim=1)

                preds = torch.argmax(probs, dim=1)

                # ------------------------------------------
                # Store results
                # ------------------------------------------

                self.targets.extend(
                    labels.cpu().numpy().tolist()
                )

                self.predictions.extend(
                    preds.cpu().numpy().tolist()
                )

                self.probabilities.extend(
                    probs.cpu().numpy()
                )

                self.image_ids.extend(
                    image_ids.cpu().numpy().tolist()
                )

        # --------------------------------------------------
        # Compute average loss
        # --------------------------------------------------

        self.loss = (
            running_loss /
            len(self.dataloader.dataset)
        )

        # --------------------------------------------------
        # Convert probabilities to numpy array
        # --------------------------------------------------

        self.probabilities = np.asarray(
            self.probabilities
        )

        # --------------------------------------------------
        # Compute evaluation metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            self.targets,
            self.predictions,
        )

        precision = precision_score(
            self.targets,
            self.predictions,
            average="binary",
            zero_division=0,
        )

        recall = recall_score(
            self.targets,
            self.predictions,
            average="binary",
            zero_division=0,
        )

        f1 = f1_score(
            self.targets,
            self.predictions,
            average="binary",
            zero_division=0,
        )

        # --------------------------------------------------
        # Store metrics
        # --------------------------------------------------

        self.metrics = {
            "loss": self.loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # --------------------------------------------------
        # Build results dataframe
        # --------------------------------------------------

        self.results_df = pd.DataFrame({
                                        "image_id": np.asarray(self.image_ids, dtype=np.int64),
                                        "true_label": np.asarray(self.targets, dtype=np.int64),
                                        "prediction": np.asarray(self.predictions, dtype=np.int64),
                                        "prob_uninfected": self.probabilities[:, 0],
                                        "prob_parasitized": self.probabilities[:, 1],
                                        "confidence": np.max(self.probabilities, axis=1),
                                        })

        return self.metrics

    def get_results(self):
        """
        Return the evaluation results DataFrame.

        Returns
        -------
        pandas.DataFrame
        """

        return self.results_df.copy()

    def classification_report(self):
        """
        Generate the sklearn classification report.

        Returns
        -------
        str
            Classification report.
        """

        report = classification_report(
            self.targets,
            self.predictions,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )

        return report

    def confusion_matrix(self, normalize=None):
        """
        Plot the confusion matrix.

        Parameters
        ----------
        normalize : str, optional
            None, "true", "pred", or "all".

        Returns
        -------
        matplotlib.figure.Figure
        """

        cm = confusion_matrix(
            self.targets,
            self.predictions,
            normalize=normalize,
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names,
        )

        disp.plot(
            cmap="Blues",
            values_format=".2f" if normalize else "d",
            ax=ax,
            colorbar=False,
        )

        ax.set_title("Confusion Matrix")

        plt.tight_layout()

        return fig

    def roc_curve(self):
        """
        Plot the ROC curve.

        Returns
        -------
        matplotlib.figure.Figure
        """

        fpr, tpr, _ = roc_curve(
            self.targets,
            self.probabilities[:, 1],
        )

        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"AUC = {roc_auc:.4f}",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        plt.tight_layout()

        return fig

    def precision_recall_curve(self):
        """
        Plot the Precision-Recall curve.

        Returns
        -------
        matplotlib.figure.Figure
        """

        precision, recall, _ = precision_recall_curve(
            self.targets,
            self.probabilities[:, 1],
        )

        ap = average_precision_score(
            self.targets,
            self.probabilities[:, 1],
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"AP = {ap:.4f}",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()

        plt.tight_layout()

        return fig

    def sample_predictions(
    self,
    dataframe,
    n=9,
    random_state=42,
    ):
        """
        Display randomly selected test predictions.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Test dataframe containing image metadata.

        n : int
            Number of images to display.

        random_state : int
            Random seed.

        Returns
        -------
        matplotlib.figure.Figure
        """

        rng = np.random.default_rng(random_state)

        samples = self.results_df.sample(
            n=min(n, len(self.results_df)),
            random_state=random_state,
        )

        cols = 3
        rows = int(np.ceil(len(samples) / cols))

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(12, 4 * rows),
        )

        axes = np.array(axes).reshape(-1)

        for ax, prediction in zip(axes, samples.to_dict("records")):

            image_info = dataframe[
                dataframe["image_id"] == prediction["image_id"]
            ].iloc[0]

            image = Image.open(image_info["filepath"]).convert("RGB")

            ax.imshow(image)
            ax.axis("off")

            true_label = int(prediction["true_label"])
            pred_label = int(prediction["prediction"])

            ax.set_title(
                f"True : {self.class_names[true_label]}\n"
                f"Pred : {self.class_names[pred_label]}\n"
                f"Conf : {prediction['confidence']:.2%}",
                fontsize=10,
            )

        # Hide unused axes
        for ax in axes[len(samples):]:
            ax.axis("off")

        plt.tight_layout()

        return fig

    def save_results(self):
        """
        Save evaluation results to disk.

        Files Saved
        -----------
        evaluation_metrics.csv
        evaluation_predictions.csv
        classification_report.txt
        """

        # ----------------------------------------
        # Metrics
        # ----------------------------------------

        metrics_df = pd.DataFrame(
            [self.metrics]
        )

        metrics_df.to_csv(
            self.output_dir / "evaluation_metrics.csv",
            index=False,
        )

        # ----------------------------------------
        # Predictions
        # ----------------------------------------

        self.results_df.to_csv(
            self.output_dir / "evaluation_predictions.csv",
            index=False,
        )

        # ----------------------------------------
        # Classification Report
        # ----------------------------------------

        with open(
            self.output_dir / "classification_report.txt",
            "w",
        ) as f:

            f.write(
                self.classification_report()
            )

        print(
            f"Evaluation results saved to:\n"
            f"{self.output_dir}"
        )

    def summary(self):
        """
        Print a summary of the evaluation results.
        """

        print("=" * 60)
        print("              Model Evaluation Summary")
        print("=" * 60)

        print(f"Loss       : {self.metrics['loss']:.4f}")
        print(f"Accuracy   : {self.metrics['accuracy']:.4f}")
        print(f"Precision  : {self.metrics['precision']:.4f}")
        print(f"Recall     : {self.metrics['recall']:.4f}")
        print(f"F1 Score   : {self.metrics['f1']:.4f}")

        print()

        print(f"Test Samples : {len(self.results_df)}")

        print(
            f"Output Directory : {self.output_dir}"
        )

        print("=" * 60)