# AlexNet-Style Deep CNN on MNIST using TensorFlow Datasets (TFDS)

## Overview
This project presents a **research-grade implementation of an AlexNet-inspired deep convolutional neural network (CNN)** trained on the **MNIST dataset**, leveraging **TensorFlow Datasets (TFDS)** for scalable data handling and clean dataset splits. The work intentionally applies a **high-capacity vision architecture**—originally designed for ImageNet—to a low-resolution digit recognition task to analyze architectural generalization, preprocessing pipelines, and training dynamics.

This project is suitable for **graduate portfolios**, demonstrating deep learning rigor, data engineering maturity, and experimental clarity.

---

## Objectives
- Study the behavior of **large-scale CNN architectures** on simple datasets
- Demonstrate **end-to-end TFDS pipelines** with custom preprocessing
- Analyze overparameterization, convergence, and class-wise performance
- Establish a reusable framework for **vision model scalability research**

---

## Dataset
- **Dataset:** MNIST (via TensorFlow Datasets)
- **Total samples:** 60,000
- **Splits:**
  - Training: 80%
  - Validation: 10%
  - Testing: 10%
- **Labels:** Digits 0–9

---

## Data Preprocessing Pipeline
The preprocessing is implemented using TensorFlow layers and dataset mapping:

1. **Resizing:**  
   - Original 28×28 images resized to **227×227**
2. **Normalization:**  
   - Pixel values scaled to `[0,1]`
3. **Channel Expansion:**  
   - Grayscale images converted to **3-channel RGB**
4. **Batching:**  
   - Mini-batches of size 32 for efficient training

This pipeline mimics **ImageNet-style input formatting**, enabling direct architectural reuse.

---

## Model Architecture
The network closely follows an **AlexNet-inspired design**, consisting of:

### Convolutional Feature Extractor
- Large-kernel initial convolution (11×11, stride 4)
- Progressive depth expansion (96 → 256 → 384)
- Multiple convolutional stages with ReLU activations
- MaxPooling layers for spatial downsampling

### Fully Connected Classifier
- Two dense layers with 4096 neurons
- Dropout regularization (0.4) to mitigate overfitting
- High-capacity intermediate representation (1000 neurons)
- Softmax output for 10-class classification

This architecture intentionally exceeds MNIST’s complexity to evaluate **representation learning under architectural mismatch**.

---

## Training Configuration
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Batch Size:** 32
- **Epochs:** Up to 5
- **Regularization:** Dropout + EarlyStopping
- **Callback:** EarlyStopping on training accuracy

---

## Evaluation Strategy
The model is evaluated using:
- Overall classification accuracy
- Training and validation loss curves
- Probability-based predictions
- Confusion matrix visualization for class-wise performance

---

## Results & Key Observations
- The model converges reliably despite heavy overparameterization
- Demonstrates strong feature learning even on simple digit data
- Highlights the trade-off between **model capacity and dataset simplicity**
- Confusion matrix confirms consistent digit discrimination

---

## Visual Outputs Included
- Model architecture diagram
- Training & validation accuracy/loss plots
- Confusion matrix heatmap

These visualizations support **transparent model diagnostics and interpretability**.

---

## Research Significance
This project demonstrates:
- Advanced CNN architecture engineering
- TFDS-based scalable data pipelines
- Controlled experimentation on model complexity
- Best practices in evaluation and visualization

It provides a strong foundation for:
- Transfer learning studies
- Architecture ablation experiments
- Vision model benchmarking research

---

## Future Extensions
- Replace MNIST with CIFAR-10 or medical imaging datasets
- Introduce Batch Normalization and learning rate scheduling
- Compare against lightweight CNN baselines
- Integrate pretrained ImageNet weights

---

## Technology Stack
- Python
- TensorFlow / Keras
- TensorFlow Datasets (TFDS)
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## Author Note
This repository is designed for **research-oriented academic and industrial evaluation**, showcasing deep learning architecture design, data engineering discipline, and experimental clarity.
