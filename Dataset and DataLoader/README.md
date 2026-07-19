# Dataset and DataLoader - PyTorch

## Overview

This project demonstrates how to create a **custom Dataset** and use a **DataLoader** in PyTorch. These are essential components for efficiently loading and batching data during deep learning model training.

A simple XOR dataset is used to understand the workflow before working with larger datasets such as DIV2K, CIFAR-10, or ImageNet.

## Topics Covered

* Creating a custom Dataset by inheriting from `torch.utils.data.Dataset`
* Understanding the `__init__()` method
* Implementing `__len__()`
* Implementing `__getitem__()`
* Creating a DataLoader
* Batch Processing
* Data Shuffling
* Iterating through batches
* Training a model using DataLoader
* Understanding mini-batch gradient descent

## Project Structure

```text
Dataset and DataLoader/
│
├── dataset_dataloader.py
└── README.md
```

## Dataset Workflow

```text
Dataset
   │
   ▼
Stores Data
   │
   ▼
Returns One Sample (__getitem__)
   │
   ▼
DataLoader
   │
   ▼
Creates Batches
   │
   ▼
Neural Network
```

## Training Pipeline

```text
Dataset
   │
   ▼
DataLoader
   │
   ▼
Batch of Features & Labels
   │
   ▼
Model
   │
   ▼
Prediction
   │
   ▼
Loss Function
   │
   ▼
Backward Pass
   │
   ▼
Optimizer
   │
   ▼
Updated Weights
```

## Concepts Learned

During this practice, I learned:

* Why custom datasets are required in PyTorch.
* The purpose of `__init__()`, `__len__()`, and `__getitem__()`.
* How DataLoader creates mini-batches.
* The importance of `batch_size`.
* Why `shuffle=True` helps improve model training.
* How mini-batch training differs from training on the entire dataset.
* How DataLoader integrates with the neural network training loop.

