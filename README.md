# Malaria Cell Classification

A deep learning system for classifying microscopic blood-cell images as **Parasitized** or **Uninfected**, with model confidence and **Grad-CAM visual explanations**.

The project covers the complete machine-learning workflow from data understanding and preprocessing through CNN training, evaluation, model interpretation, and deployment as a public Streamlit web application.

## 🚀 Live Demo

**Web Application:**
(https://malaria-cell-classification-kdnctzfqd2cw864xu22n8k.streamlit.app/)

The deployed application allows a user to upload a malaria cell image and receive:

* Predicted class
* Prediction confidence
* Grad-CAM visualization showing the image regions that contributed most to the prediction

> **Important:** This application is an educational and machine-learning demonstration. It is not a medical diagnostic tool and should not be used for clinical diagnosis or treatment decisions.

---

## 📌 Project Overview

Malaria diagnosis can involve examining blood-smear images to identify malaria parasites. This project explores how a convolutional neural network (CNN) can learn visual patterns from microscopic cell images and classify them into two classes:

| Class         | Meaning                                      |
| ------------- | -------------------------------------------- |
| `Uninfected`  | Cell image without detected malaria parasite |
| `Parasitized` | Cell image containing a malaria parasite     |

The project was designed not only to train a classifier, but also to demonstrate a complete ML engineering workflow including:

* Reproducible preprocessing
* CNN model development
* Training and checkpointing
* Quantitative evaluation
* Prediction visualization
* Model explainability using Grad-CAM
* Deployment-ready inference code
* Web deployment

---

## 🧠 Model

The classifier is a configurable convolutional neural network implemented using **PyTorch**.

### Architecture

The current model uses three convolutional blocks:

```text
Input Image
    │
    ▼
Conv2D — 32 channels
    │
BatchNorm
    │
ReLU
    │
MaxPool
    │
Dropout
    │
    ▼
Conv2D — 64 channels
    │
BatchNorm
    │
ReLU
    │
MaxPool
    │
Dropout
    │
    ▼
Conv2D — 128 channels
    │
BatchNorm
    │
ReLU
    │
MaxPool
    │
Dropout
    │
    ▼
Adaptive Average Pooling
    │
    ▼
Flatten
    │
Dropout
    │
    ▼
Linear Layer
    │
    ▼
2 Class Logits
```

### Model configuration

```python
input_channels = 3
num_classes = 2
conv_channels = [32, 64, 128]
kernel_size = 3
dropout = 0.30
activation = "relu"
use_batchnorm = True
```

The input images are RGB images resized to:

```text
128 × 128 × 3
```

---

## 📊 Dataset

The project uses the **Cell Images for Detecting Malaria** dataset.

The dataset contains microscopic blood-cell images belonging to two classes:

```text
Parasitized
Uninfected
```

The dataset is organized into class-specific directories and is processed into a structured dataframe containing image metadata and labels.

### Dataset preprocessing

The preprocessing pipeline performs:

1. Dataset discovery
2. Image validation
3. Label mapping
4. Stratified train/validation/test splitting
5. Dataset statistics calculation
6. Image resizing
7. Data augmentation for training
8. Tensor conversion
9. Channel normalization

The final image size used by the model is:

```text
128 × 128
```

### Normalization

The normalization statistics were calculated from the training dataset.

```text
Mean:
[0.5307622621, 0.4247844621, 0.4537573588]

Standard deviation:
[0.3372651461, 0.2728947299, 0.2879494788]
```

These values are stored as deployment artifacts rather than being hard-coded into the web application.

---

## 🔄 Preprocessing Pipeline

The project separates training preprocessing from deployment preprocessing while ensuring that the inference pipeline uses the same important transformations as training.

### Training

Training images receive augmentation such as:

* Random horizontal flip
* Random vertical flip
* Random rotation
* Color jitter
* Resizing
* Tensor conversion
* Normalization

### Deployment

For inference, augmentation is not applied.

Instead, the uploaded image is:

```text
Uploaded Image
      │
      ▼
RGB conversion
      │
      ▼
Resize to 128 × 128
      │
      ▼
Convert to Tensor
      │
      ▼
Normalize using training statistics
      │
      ▼
CNN
```

This prevents the deployment pipeline from introducing random transformations that could alter the prediction.

---

## 🏋️ Training

The training pipeline is implemented in PyTorch and includes:

* Training and validation loops
* Configurable optimizer
* Learning-rate scheduler
* Validation loss tracking
* Early stopping
* Model checkpointing
* Training history tracking

The best-performing model checkpoint is saved as:

```text
checkpoints/best_model.pth
```

The trained model is approximately **1.09 MB**, making it relatively lightweight for web inference.

---

## 📈 Model Evaluation

Evaluation is performed independently from the training process using the held-out test set.

The evaluation pipeline includes quantitative and visual analysis of model predictions.

Metrics implemented in the project include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Prediction confidence
* Softmax probabilities

The evaluation notebook also provides visualization of individual predictions, allowing the model's correct and incorrect classifications to be inspected visually.

See:

```text
notebooks/04_evaluation.ipynb
```

for the complete evaluation workflow.

---

## 🔍 Model Explainability with Grad-CAM

A major component of the project is **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Instead of only returning:

```text
Prediction: Parasitized
Confidence: 97%
```

the application also produces a heatmap showing the regions of the image that contributed most strongly to the prediction.

### Grad-CAM workflow

```text
Uploaded Image
      │
      ▼
CNN Forward Pass
      │
      ▼
Predicted Class
      │
      ▼
Gradient Calculation
      │
      ▼
Target Convolutional Layer
      │
      ▼
Grad-CAM Activation Map
      │
      ▼
Heatmap Overlay
      │
      ▼
User
```

The visualization helps make the model's prediction more interpretable.

### Important limitation

Grad-CAM indicates **regions that contributed to the model's prediction**. It does not prove that those regions correspond to medically meaningful parasite structures.

Therefore, the visualization should be interpreted as a model-explanation tool rather than a clinical diagnostic explanation.

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

The deployment architecture separates the training environment from the inference environment.

```text
                         GitHub
                           │
                           ▼
                    Streamlit Cloud
                           │
                           ▼
                        app.py
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Preprocessing                 Predictor
             │                           │
             └─────────────┬─────────────┘
                           ▼
                       CNN Model
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
              Prediction          Grad-CAM
                  │                 │
                  └────────┬────────┘
                           ▼
                    Streamlit UI
```

The deployed application does **not require the original training dataset**.

Instead, deployment uses:

```text
checkpoints/best_model.pth
artifacts/normalization.json
artifacts/transforms.json
```

This keeps the inference application lightweight and avoids shipping the complete training dataset to the deployment environment.

---

## 🖥️ Web Application

The Streamlit interface provides a simple workflow:

### 1. Upload an image

The user uploads a microscopic cell image.

### 2. Run inference

The image is passed through the deployment preprocessing pipeline and CNN.

### 3. Display prediction

The application displays the predicted class and confidence.

### 4. Display Grad-CAM

A Grad-CAM visualization is generated to show the image regions contributing to the prediction.

The resulting interface combines:

```text
Original Image
      +
Prediction
      +
Confidence
      +
Grad-CAM Explanation
```

---

## 📁 Project Structure

```text
Malaria-cell-classification/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── artifacts/
│   ├── normalization.json
│   └── transforms.json
│
├── checkpoints/
│   └── best_model.pth
│
├── configs/
│   └── config.py
│
├── deployment/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── predictor.py
│   ├── gradcam.py
│   └── inference.py
│
├── notebooks/
│   ├── 01_DataUnderstanding.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
│
└── src/
    ├── __init__.py
    ├── dataset.py
    ├── evaluator.py
    ├── metrics.py
    ├── model.py
    ├── preprocessing.py
    ├── trainer.py
    └── utils.py
```

---

## 🧩 Code Organization

The project is divided into two major layers.

### Training and experimentation

The `src/` package contains the components used for:

* Dataset management
* Preprocessing
* Model development
* Training
* Evaluation
* Metrics
* Utilities

### Deployment

The `deployment/` package contains only the components required for inference:

* Deployment configuration
* Inference preprocessing
* Model prediction
* Grad-CAM
* Inference service

This separation makes the deployed application independent of unnecessary training dependencies.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/calebchege/Malaria-cell-classification.git
cd Malaria-cell-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

Start Streamlit:

```bash
streamlit run app.py
```

The application will be available locally through the URL provided by Streamlit, normally:

```text
http://localhost:8501
```

---

## 📓 Notebooks

The project is organized into four main notebooks.

### `01_DataUnderstanding.ipynb`

Covers:

* Dataset inspection
* Class distribution
* Image characteristics
* Data quality checks
* Dataset exploration

### `02_preprocessing.ipynb`

Covers:

* Dataset loading
* Dataframe construction
* Train/validation/test splitting
* Dataset statistics
* Transformation pipeline
* DataLoader construction

### `03_training.ipynb`

Covers:

* Model construction
* Loss function
* Optimizer
* Scheduler
* Training
* Validation
* Checkpointing
* Training history

### `04_evaluation.ipynb`

Covers:

* Loading the trained model
* Test-set evaluation
* Classification metrics
* Confusion matrix
* Prediction confidence
* Sample predictions
* Model performance analysis

---

## 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* PyTorch
* Torchvision

### Data Science

* NumPy
* Pandas
* Scikit-learn
* Matplotlib

### Image Processing

* Pillow

### Deployment

* Streamlit
* Streamlit Community Cloud

### Version Control

* Git
* GitHub

---

## 🎯 Project Goals

The project was developed with several goals:

1. Build an end-to-end computer vision classification system.
2. Develop a configurable CNN using PyTorch.
3. Establish a reproducible preprocessing pipeline.
4. Evaluate the trained model using multiple metrics.
5. Explore model interpretability using Grad-CAM.
6. Separate training and inference code.
7. Package the trained model for deployment.
8. Deploy the system as an accessible web application.

---

## ⚠️ Limitations

This project has several important limitations.

### Dataset limitations

The model is dependent on the characteristics and quality of its training dataset. Performance on images that differ substantially from the training distribution may be lower.

### Model limitations

A CNN can learn correlations in the training data that may not correspond to medically meaningful features.

### Explainability limitations

Grad-CAM provides an indication of which image regions influenced a prediction. It does not establish causality or guarantee that the highlighted regions represent malaria parasites.

### Clinical limitation

This system has **not been validated as a clinical diagnostic system**.

It should not be used to make medical decisions, diagnose patients, or determine treatment.

---

## 🔮 Future Improvements

Potential future improvements include:

* Testing additional CNN architectures
* Transfer learning with pretrained models
* More extensive hyperparameter optimization
* Cross-dataset evaluation
* Calibration of prediction probabilities
* Robustness testing on external microscopy datasets
* Improved Grad-CAM analysis
* Additional explainability techniques
* Model quantization for lower-resource deployment
* Automated monitoring of deployed predictions
* Containerized deployment
* API-based inference
* More comprehensive model validation

---

## 📚 Learning Outcomes

This project provided practical experience across the complete machine-learning lifecycle:

```text
Data
 ↓
Exploration
 ↓
Preprocessing
 ↓
Model Development
 ↓
Training
 ↓
Evaluation
 ↓
Interpretability
 ↓
Packaging
 ↓
Deployment
 ↓
Web Application
```

It demonstrates that building a machine-learning system involves considerably more than training a model. Reliable preprocessing, evaluation, explainability, deployment architecture, and responsible communication of limitations are equally important.

---



## 📄 License
```text
MIT License
```



See the repository for the complete source code and implementation.
