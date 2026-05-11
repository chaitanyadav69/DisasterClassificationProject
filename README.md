//
# Hybrid CNN Transformer Parallel Architecture for Robust Multi-Class Disaster Image Classification

## Overview

This project implements a complete hybrid deep learning framework for multi-class disaster image classification using:

- Convolutional Neural Networks (CNNs)
- Vision Transformers (ViTs)
- Multi-Head Self Attention (MHSA)
- TensorFlow/Keras
- Tkinter GUI
- TensorFlow Lite
- ONNX Export

The system classifies disaster-related images into 12 fine-grained disaster and non-disaster categories using three hybrid architectures:

1. CNN → Transformer (Sequential)
2. Transformer → CNN (Hierarchical)
3. CNN + Transformer (Parallel Proposed Model)

The proposed Parallel CNN+Transformer architecture achieves the best performance by independently learning local spatial features and global contextual relationships before feature fusion.

---

# Features

- Multi-class disaster image classification
- Hybrid CNN + Transformer architectures
- TensorFlow/Keras implementation
- Tkinter GUI for prediction
- TensorFlow Lite export
- ONNX export support
- Modular architecture
- Edge deployment support
- Data augmentation pipeline
- Model evaluation metrics

---

# Disaster Classes

The system classifies images into the following 12 classes:

1. Earthquake Damage
2. Infrastructure Damage
3. Urban Fire
4. Wildfire
5. Human Damage
6. Drought
7. Landslide
8. Non Damage Human
9. Non Damage Buildings
10. Non Damage Forest
11. Non Damage Sea
12. Water Disaster

---

# Proposed Architecture

## Sequential Architecture
CNN extracts local spatial features first, followed by Transformer-based global context modeling.

## Hierarchical Architecture
Transformer captures global context first, followed by CNN-based local refinement.

## Parallel Architecture (Proposed)
CNN and Transformer branches process the same image independently and fuse their outputs before classification.

Advantages:
- Eliminates inter-branch information bottlenecks
- Preserves local and global features independently
- Improves classification performance

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| TensorFlow/Keras | Deep Learning Framework |
| NumPy | Numerical Computation |
| OpenCV | Image Processing |
| Pillow | Image Loading |
| Tkinter | GUI Development |
| Matplotlib | Visualization |
| scikit-learn | Evaluation Metrics |
| TensorFlow Lite | Edge Deployment |
| ONNX | Cross-platform Model Export |

---

# System Requirements

## Minimum Requirements

- Intel i5 Processor
- 8 GB RAM
- 10 GB Free Storage
- Python 3.10+
- Windows/Linux/macOS

## Recommended Requirements

- NVIDIA GPU
- CUDA/cuDNN
- 16 GB RAM
- SSD Storage

---

# Folder Structure

```text
DisasterClassificationProject/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── training/
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── evaluation.py
│
├── gui/
│   └── app.py
│
├── models/
│   ├── sequential_model.h5
│   ├── hierarchical_model.h5
│   └── parallel_model.h5
│
├── exports/
│
├── requirements.txt
├── labels.txt
└── README.md
