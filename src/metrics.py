from pathlib import Path

import pandas as pd


class MetricTracker:
    """
    Tracks and manages training and validation metrics
    throughout the training process.

    Metrics tracked
    ---------------
    - Training Loss
    - Validation Loss
    - Training Accuracy
    - Validation Accuracy
    - Training Precision
    - Validation Precision
    - Training Recall
    - Validation Recall
    - Training F1 Score
    - Validation F1 Score
    """

    def __init__(self):
        """
        Initialize an empty metric history.
        """

        self.history = {
            "train_loss": [],
            "val_loss": [],

            "train_accuracy": [],
            "val_accuracy": [],

            "train_precision": [],
            "val_precision": [],

            "train_recall": [],
            "val_recall": [],

            "train_f1": [],
            "val_f1": [],
        }

    def update(self, **kwargs: float):
        """
        Update the metric history.

        Parameters
        ----------
        **kwargs : float
            Metric name and corresponding value.

        Example
        -------
        tracker.update(
            train_loss=0.42,
            val_loss=0.38,
            train_accuracy=0.91,
            val_accuracy=0.89,
        )
        """

        for key, value in kwargs.items():

            if key not in self.history:
                raise KeyError(f"Unknown metric: '{key}'")

            self.history[key].append(float(value))

    def reset(self):
        """
        Clear all stored metrics.
        """

        for key in self.history:
            self.history[key].clear()

    def get_history(self):
        """
        Return the complete metric history.

        Returns
        -------
        dict
            Dictionary containing all tracked metrics.
        """

        return self.history

    def to_dataframe(self):
        """
        Convert the metric history to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
        """

        return pd.DataFrame(self.history)

    def save(self, filepath):
        """
        Save the training history as a CSV file.

        Parameters
        ----------
        filepath : str or Path
            Destination CSV file.
        """

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.to_dataframe().to_csv(
            filepath,
            index=False,
        )

    def __len__(self):
        """
        Return the number of completed epochs.

        Returns
        -------
        int
        """

        return len(self.history["train_loss"])

    def __repr__(self):
        """
        String representation of the tracker.
        """

        return (
            f"{self.__class__.__name__}"
            f"(epochs_recorded={len(self)})"
        )