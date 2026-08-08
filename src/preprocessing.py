from pathlib import Path
import zipfile
import json
import os

import numpy as np
import pandas as pd

from PIL import Image

import torch

from torch.utils.data import DataLoader,Dataset

from torchvision import transforms

from sklearn.model_selection import train_test_split
from .dataset import MalariaDataset





class MalariaDataPreprocessor:
    """
    Handles preprocessing of the malaria cell image dataset.
    """

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    CLASS_MAPPING = {
        "Parasitized": 1,
        "Uninfected": 0,
    }

    def __init__(
        self,
        project_dir,
        image_size=(128, 128),
        batch_size=32,
        validation_split=0.15,
        test_split=0.15,
        random_state=42,
        num_workers=0,
        pin_memory=True,

        # Data augmentation
        rotation_degrees=15,
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.5,
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
    ):
        """
        Parameters
        ----------
        project_dir : str or Path
            Root directory of the project.
        """

        # --------------------------------------------------
        # Project paths
        # --------------------------------------------------

        self.project_dir = Path(project_dir)

        self.dataset_dir = self.project_dir / "dataset"

        self.dataset_zip = self.dataset_dir / "data.zip"

        self.extracted_dir = self.dataset_dir / "extracted"

        self.cell_images_path = (
            self.extracted_dir / "cell_images"
        )

        self.artifacts_dir = (
            self.project_dir / "artifacts"
        )

        self.checkpoints_dir = (
            self.project_dir / "checkpoints"
        )

        self.outputs_dir = (
            self.project_dir / "outputs"
        )

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        self.image_size = image_size

        self.batch_size = batch_size

        self.validation_split = validation_split

        self.test_split = test_split

        self.random_state = random_state

        self.num_workers = num_workers

        self.pin_memory = pin_memory

        # --------------------------------------------------
        # Dataset metadata
        # --------------------------------------------------

        self.class_paths = {}

        self.data = None

        self.train_df = None
        self.val_df = None
        self.test_df = None

        # --------------------------------------------------
        # Normalization statistics
        # --------------------------------------------------

        self.mean = None

        self.std = None

        # --------------------------------------------------
        # Augmentation configuration
        # --------------------------------------------------

        self.rotation_degrees = rotation_degrees

        self.horizontal_flip_prob = horizontal_flip_prob

        self.vertical_flip_prob = vertical_flip_prob

        self.brightness = brightness

        self.contrast = contrast

        self.saturation = saturation

        # --------------------------------------------------
        # Image transforms
        # --------------------------------------------------

        self.train_transform = None

        self.eval_transform = None

        # --------------------------------------------------
        # PyTorch datasets
        # --------------------------------------------------

        self.train_dataset = None

        self.val_dataset = None

        self.test_dataset = None

        # --------------------------------------------------
        # DataLoaders
        # --------------------------------------------------

        self.train_loader = None

        self.val_loader = None

        self.test_loader = None
    def prepare_dataset(self):
        """
        Prepare the dataset for preprocessing.

        Workflow
        --------
        1. Verify the project structure.
        2. Verify that data.zip exists.
        3. Extract the archive if necessary.
        4. Validate the dataset structure.
        5. Build the class path dictionary.

        Returns
        -------
        self
        """

        import zipfile

        # --------------------------------------------------
        # Verify project directory
        # --------------------------------------------------

        if not self.project_dir.exists():

            raise FileNotFoundError(
                f"Project directory not found:\n"
                f"{self.project_dir}"
            )

        # --------------------------------------------------
        # Create required directories
        # --------------------------------------------------

        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoints_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.outputs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.extracted_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Verify dataset archive
        # --------------------------------------------------

        if not self.dataset_zip.exists():

            raise FileNotFoundError(
                f"Dataset archive not found:\n"
                f"{self.dataset_zip}"
            )

        # --------------------------------------------------
        # Extract dataset only if necessary
        # --------------------------------------------------

        if not self.cell_images_path.exists():

            print("=" * 60)
            print("Extracting dataset...")
            print("=" * 60)

            with zipfile.ZipFile(
                self.dataset_zip,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    self.extracted_dir
                )

            print("Extraction complete.\n")

        else:

            print("Dataset already extracted.\n")

        # --------------------------------------------------
        # Validate extracted structure
        # --------------------------------------------------

        if not self.cell_images_path.exists():

            raise FileNotFoundError(
                f"'cell_images' directory not found:\n"
                f"{self.cell_images_path}"
            )

        # --------------------------------------------------
        # Validate class folders
        # --------------------------------------------------

        self.class_paths = {}

        for class_name in self.CLASS_MAPPING:

            class_path = (
                self.cell_images_path / class_name
            )

            if not class_path.exists():

                raise FileNotFoundError(
                    f"Missing class folder:\n"
                    f"{class_path}"
                )

            self.class_paths[class_name] = class_path

        print("Dataset validation successful.")

        print(f"Dataset Location : {self.cell_images_path}")

        print(f"Classes Found    : {list(self.class_paths.keys())}")

        return self

    
    
    def build_dataframe(self):
        """
        Build a metadata DataFrame for all valid images.

        The resulting dataframe contains one row per image.

        Columns
        -------
        image_id
        filepath
        filename
        class_name
        label
        width
        height
        channels

        Returns
        -------
        self
        """

        records = []

        image_id = 0

        for class_name, class_path in self.class_paths.items():

            label = self.CLASS_MAPPING[class_name]

            # Sort files for reproducibility
            image_paths = sorted(class_path.iterdir())

            for image_path in image_paths:

                if image_path.suffix.lower() not in self.VALID_EXTENSIONS:
                    continue

                try:
                    with Image.open(image_path) as img:

                        width, height = img.size

                        channels = len(img.getbands())

                except Exception:

                    print(f"Skipping unreadable image: {image_path}")

                    continue

                records.append({

                    "image_id": image_id,

                    "filepath": str(image_path),

                    "filename": image_path.name,

                    "class_name": class_name,

                    "label": label,

                    "width": width,

                    "height": height,

                    "channels": channels,

                })

                image_id += 1

        self.data = pd.DataFrame(records)

        if self.data.empty:

            raise ValueError(
                "No valid image files were found."
            )

        print("=" * 60)
        print("Dataset Metadata")
        print("=" * 60)

        print(f"Total Images : {len(self.data)}")

        print()

        print(self.data.head())

        return self

    def split_data(self):
        """
        Split the dataset into training,
        validation and testing sets.

        Returns
        -------
        self
        """

        if self.data is None:

            raise ValueError(
                "Dataset dataframe has not been built."
            )

        if self.validation_split + self.test_split >= 1:

            raise ValueError(
                "validation_split + test_split must be less than 1."
            )

        # ------------------------------------------
        # First split
        # ------------------------------------------

        train_df, temp_df = train_test_split(

            self.data,

            test_size=self.validation_split + self.test_split,

            stratify=self.data["label"],

            shuffle=True,

            random_state=self.random_state,

        )

        # ------------------------------------------
        # Relative validation size
        # ------------------------------------------

        validation_fraction = (

            self.validation_split

            /

            (self.validation_split + self.test_split)

        )

        # ------------------------------------------
        # Second split
        # ------------------------------------------

        val_df, test_df = train_test_split(

            temp_df,

            test_size=1 - validation_fraction,

            stratify=temp_df["label"],

            shuffle=True,

            random_state=self.random_state,

        )

        self.train_df = train_df.reset_index(drop=True)

        self.val_df = val_df.reset_index(drop=True)

        self.test_df = test_df.reset_index(drop=True)

        self._validate_splits()

        print("=" * 60)
        print("Dataset Split")
        print("=" * 60)

        print(f"Training   : {len(self.train_df)}")

        print(f"Validation : {len(self.val_df)}")

        print(f"Testing    : {len(self.test_df)}")

        return self


    def _validate_splits(self):
        """
        Validate dataset splits.

        Checks
        ------
        1. No overlap.
        2. No missing samples.
        3. Class balance preserved.

        Returns
        -------
        bool
        """

        train_paths = set(self.train_df["filepath"])

        val_paths = set(self.val_df["filepath"])

        test_paths = set(self.test_df["filepath"])

        # ------------------------------------------
        # Overlap
        # ------------------------------------------

        if train_paths & val_paths:

            raise ValueError(
                "Training and validation sets overlap."
            )

        if train_paths & test_paths:

            raise ValueError(
                "Training and testing sets overlap."
            )

        if val_paths & test_paths:

            raise ValueError(
                "Validation and testing sets overlap."
            )

        # ------------------------------------------
        # Missing samples
        # ------------------------------------------

        all_paths = train_paths | val_paths | test_paths

        if len(all_paths) != len(self.data):

            raise ValueError(
                "Split validation failed."
            )

        # ------------------------------------------
        # Class distribution
        # ------------------------------------------

        print("\nClass Distribution")

        print("-" * 40)

        distribution = pd.DataFrame({

            "Train": self.train_df["class_name"].value_counts(),

            "Validation": self.val_df["class_name"].value_counts(),

            "Test": self.test_df["class_name"].value_counts(),

        }).fillna(0).astype(int)

        print(distribution)

        print("\nSplit validation successful.\n")

        return True

    def compute_dataset_statistics(self, force_recompute=False):
        """
        Compute or load the RGB normalization statistics.

        Statistics are computed from the training set only.

        If normalization statistics already exist in the artifacts
        directory, they are loaded unless force_recompute=True.

        Parameters
        ----------
        force_recompute : bool, default=False
            If True, recompute statistics even if cached values exist.

        Returns
        -------
        self
        """

        import json

        statistics_file = (
            self.artifacts_dir / "normalization.json"
        )

        # --------------------------------------------------
        # Load cached statistics
        # --------------------------------------------------

        if statistics_file.exists() and not force_recompute:

            print("Loading normalization statistics...")

            with open(statistics_file, "r") as f:
                statistics = json.load(f)

            self.mean = np.array(statistics["mean"])

            self.std = np.array(statistics["std"])

            print("Normalization statistics loaded.")

            return self

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if self.train_df is None:

            raise ValueError(
                "Training dataframe not found.\n"
                "Call split_data() first."
            )

        print("=" * 60)
        print("Computing Dataset Statistics")
        print("=" * 60)

        channel_sum = np.zeros(3, dtype=np.float64)

        channel_squared_sum = np.zeros(3, dtype=np.float64)

        total_pixels = 0

        # --------------------------------------------------
        # Compute statistics
        # --------------------------------------------------

        for image_path in self.train_df["filepath"]:

            image = (
                Image.open(image_path)
                .convert("RGB")
                .resize(self.image_size)
            )

            image = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

            pixels = image.shape[0] * image.shape[1]

            channel_sum += image.sum(axis=(0, 1))

            channel_squared_sum += (
                image ** 2
            ).sum(axis=(0, 1))

            total_pixels += pixels

        self.mean = channel_sum / total_pixels

        self.std = np.sqrt(

            channel_squared_sum / total_pixels

            - self.mean ** 2

        )

        # --------------------------------------------------
        # Save statistics
        # --------------------------------------------------

        statistics = {

            "image_size": list(self.image_size),

            "mean": self.mean.tolist(),

            "std": self.std.tolist(),

            "num_training_images": len(self.train_df),

            "total_pixels": int(total_pixels),

        }

        with open(statistics_file, "w") as f:

            json.dump(
                statistics,
                f,
                indent=4,
            )

        print()

        print("Normalization statistics computed.")

        print()

        print(f"Mean : {self.mean.round(6)}")

        print(f"Std  : {self.std.round(6)}")

        print()

        print(f"Saved to:\n{statistics_file}")

        return self



    def _save_transform_configuration(self):
        """
        Save the image preprocessing configuration.

        The configuration is stored in the artifacts directory
        to ensure that training, evaluation and inference use
        identical preprocessing settings.

        Returns
        -------
        None
        """

        import json

        transform_config = {

            "image_size": list(self.image_size),

            "normalization": {
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
            },

            "augmentation": {

                "rotation_degrees": self.rotation_degrees,

                "horizontal_flip_probability":
                    self.horizontal_flip_prob,

                "vertical_flip_probability":
                    self.vertical_flip_prob,

                "brightness":
                    self.brightness,

                "contrast":
                    self.contrast,

                "saturation":
                    self.saturation,
            }
        }

        save_path = (
            self.artifacts_dir /
            "transforms.json"
        )

        with open(save_path, "w") as f:

            json.dump(
                transform_config,
                f,
                indent=4,
            )

        print(f"Transform configuration saved to:\n{save_path}")


    def build_transforms(self):
        """
        Build image transformation pipelines for training and evaluation.

        Returns
        -------
        self
        """

        if self.mean is None or self.std is None:

            raise ValueError(
                "Normalization statistics not found.\n"
                "Call compute_dataset_statistics() first."
            )

        self.train_transform = transforms.Compose([

            transforms.Resize(
                self.image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),

            transforms.CenterCrop(
                self.image_size
            ),

            transforms.RandomHorizontalFlip(
                p=self.horizontal_flip_prob
            ),

            transforms.RandomVerticalFlip(
                p=self.vertical_flip_prob
            ),

            transforms.RandomRotation(
                degrees=self.rotation_degrees
            ),

            transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=self.mean.tolist(),
                std=self.std.tolist(),
            ),
        ])

        self.eval_transform = transforms.Compose([

            transforms.Resize(
                self.image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),

            transforms.CenterCrop(
                self.image_size
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=self.mean.tolist(),
                std=self.std.tolist(),
            ),
        ])

        # Save preprocessing configuration
        self._save_transform_configuration()

        print("Image transforms created.")

        return self
    
    def build_datasets(self):
        """
        Build PyTorch Dataset objects.

        Returns
        -------
        self
        """

        if self.train_df is None:

            raise ValueError(
                "Training dataframe not available."
            )

        if self.train_transform is None:

            raise ValueError(
                "Transforms have not been built."
            )

        self.train_dataset = MalariaDataset(

            dataframe=self.train_df,

            transform=self.train_transform,

        )

        self.val_dataset = MalariaDataset(

            dataframe=self.val_df,

            transform=self.eval_transform,

        )

        self.test_dataset = MalariaDataset(

            dataframe=self.test_df,

            transform=self.eval_transform,

        )

        print()

        print("Datasets created")

        print(f"Train      : {len(self.train_dataset)}")

        print(f"Validation : {len(self.val_dataset)}")

        print(f"Test       : {len(self.test_dataset)}")

        return self


    def build_dataloaders(self):
        """
        Build PyTorch DataLoaders.

        Returns
        -------
        self
        """

        if self.train_dataset is None:

            raise ValueError(
                "Datasets have not been created."
            )

        pin_memory = (
            self.pin_memory
            and torch.cuda.is_available()
        )

        persistent_workers = (
            self.num_workers > 0
        )

        self.train_loader = DataLoader(

            self.train_dataset,

            batch_size=self.batch_size,

            shuffle=True,

            num_workers=self.num_workers,

            pin_memory=pin_memory,

            persistent_workers=persistent_workers,

        )

        self.val_loader = DataLoader(

            self.val_dataset,

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=self.num_workers,

            pin_memory=pin_memory,

            persistent_workers=persistent_workers,

        )

        self.test_loader = DataLoader(

            self.test_dataset,

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=self.num_workers,

            pin_memory=pin_memory,

            persistent_workers=persistent_workers,

        )

        print()

        print("DataLoaders created")

        print(f"Batch Size : {self.batch_size}")

        print(f"Workers    : {self.num_workers}")

        print(f"Pin Memory : {pin_memory}")

        return self

    def summary(self):
        """
        Print a summary of the preprocessing pipeline.

        Returns
        -------
        self
        """

        print("\n" + "=" * 70)
        print("           MALARIA DATA PREPROCESSING SUMMARY")
        print("=" * 70)

        # --------------------------------------------------
        # Project
        # --------------------------------------------------

        print("\nPROJECT")
        print("-" * 70)

        print(f"Project Directory : {self.project_dir}")
        print(f"Dataset Archive   : {self.dataset_zip}")
        print(f"Dataset Location  : {self.cell_images_path}")
        print(f"Artifacts         : {self.artifacts_dir}")
        print(f"Outputs           : {self.outputs_dir}")
        print(f"Checkpoints       : {self.checkpoints_dir}")

        # --------------------------------------------------
        # Dataset
        # --------------------------------------------------

        print("\nDATASET")
        print("-" * 70)

        print(f"Total Images      : {len(self.data)}")
        print(f"Training Images   : {len(self.train_df)}")
        print(f"Validation Images : {len(self.val_df)}")
        print(f"Testing Images    : {len(self.test_df)}")

        print("\nClass Distribution")

        distribution = pd.DataFrame({

            "Train": self.train_df["class_name"].value_counts(),

            "Validation": self.val_df["class_name"].value_counts(),

            "Test": self.test_df["class_name"].value_counts(),

        }).fillna(0).astype(int)

        print(distribution)

        # --------------------------------------------------
        # Image configuration
        # --------------------------------------------------

        print("\nIMAGE CONFIGURATION")
        print("-" * 70)

        print(f"Image Size : {self.image_size}")
        print(f"Batch Size : {self.batch_size}")

        # --------------------------------------------------
        # Normalization
        # --------------------------------------------------

        print("\nNORMALIZATION")
        print("-" * 70)

        print(f"Mean : {self.mean.round(6).tolist()}")
        print(f"Std  : {self.std.round(6).tolist()}")

        # --------------------------------------------------
        # Augmentation
        # --------------------------------------------------

        print("\nAUGMENTATION")
        print("-" * 70)

        print(f"Rotation          : ±{self.rotation_degrees}°")
        print(f"Horizontal Flip   : {self.horizontal_flip_prob}")
        print(f"Vertical Flip     : {self.vertical_flip_prob}")
        print(f"Brightness        : {self.brightness}")
        print(f"Contrast          : {self.contrast}")
        print(f"Saturation        : {self.saturation}")

        # --------------------------------------------------
        # DataLoader
        # --------------------------------------------------

        print("\nDATALOADER")
        print("-" * 70)

        print(f"Batch Size : {self.batch_size}")
        print(f"Workers    : {self.num_workers}")
        print(f"Pin Memory : {self.pin_memory}")

        # --------------------------------------------------
        # Artifacts
        # --------------------------------------------------

        print("\nGENERATED ARTIFACTS")
        print("-" * 70)

        normalization_file = self.artifacts_dir / "normalization.json"
        transforms_file = self.artifacts_dir / "transforms.json"

        print(f"Normalization : {'✓' if normalization_file.exists() else '✗'}")
        print(f"Transforms    : {'✓' if transforms_file.exists() else '✗'}")

        # --------------------------------------------------
        # Pipeline status
        # --------------------------------------------------

        print("\nPIPELINE STATUS")
        print("-" * 70)

        print(f"Dataset Prepared      : {'✓' if self.cell_images_path.exists() else '✗'}")
        print(f"Metadata Built        : {'✓' if self.data is not None else '✗'}")
        print(f"Dataset Split         : {'✓' if self.train_df is not None else '✗'}")
        print(f"Statistics Computed   : {'✓' if self.mean is not None else '✗'}")
        print(f"Transforms Built      : {'✓' if self.train_transform is not None else '✗'}")
        print(f"Datasets Built        : {'✓' if self.train_dataset is not None else '✗'}")
        print(f"DataLoaders Built     : {'✓' if self.train_loader is not None else '✗'}")

        print("=" * 70)

        return self

    def prepare(self):
        """
        Execute the complete preprocessing pipeline.

        Returns
        -------
        self
        """

        (
            self
            .prepare_dataset()
            .build_dataframe()
            .split_data()
            .compute_dataset_statistics()
            .build_transforms()
            .build_datasets()
            .build_dataloaders()
            .summary()
        )

        return self