## Overview

This project demonstrates the fundamentals of building and training a **Multi-Layer Perceptron (MLP)** using **PyTorch**. The objective is to understand how a neural network performs forward propagation, calculates loss, computes gradients using Autograd, and updates its parameters through an optimizer.

The model is trained on the **XOR dataset**, a classic problem that cannot be solved by a single perceptron but can be learned by an MLP with a hidden layer.

## Topics Covered

* Creating a simple dataset
* Building an MLP using `nn.Sequential`
* Understanding `nn.Linear`
* ReLU Activation Function
* Sigmoid Activation Function
* Binary Cross Entropy Loss (`BCELoss`)
* Stochastic Gradient Descent (SGD) Optimizer
* Forward Pass
* Backward Pass (`loss.backward()`)
* Weight Updates (`optimizer.step()`)
* Making Predictions
* Understanding Trainable Parameters

## Model Architecture

```text
Input (2 Features)
        │
        ▼
Linear Layer (2 → 4)
        │
        ▼
ReLU
        │
        ▼
Linear Layer (4 → 1)
        │
        ▼
Sigmoid
        │
        ▼
Prediction
```

## Dataset

The network is trained on the XOR dataset:

| Input  | Target |
| ------ | ------ |
| (0, 0) | 0      |
| (0, 1) | 1      |
| (1, 0) | 1      |
| (1, 1) | 0      |

## Training Pipeline

```text
Input Data
     │
     ▼
Forward Pass
     │
     ▼
Prediction
     │
     ▼
Loss Calculation
     │
     ▼
Backward Pass
     │
     ▼
Optimizer Updates Weights
     │
     ▼
Repeat
```

## Learning Outcomes

After completing this practice, I understood:

* How an MLP is constructed in PyTorch.
* The purpose of hidden and output layers.
* How tensors move through a neural network.
* How the loss function measures prediction error.
* How Autograd computes gradients automatically.
* How the optimizer updates weights using gradients.
* Why neural networks improve over multiple training epochs.
* The importance of reproducibility using `torch.manual_seed()`.

