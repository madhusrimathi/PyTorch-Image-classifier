# PyTorch Image Classifier

A from-scratch convolutional neural network built with **PyTorch** to classify images from the **CIFAR-10** dataset.

This project is intentionally small and readable: the goal is to understand the complete deep-learning workflow rather than hide it behind a high-level training framework.

## What it does

CIFAR-10 contains 32x32 colour images across 10 classes:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

The pipeline is:

```text
CIFAR-10 images
      |
      v
augmentation + normalization
      |
      v
Conv2d -> ReLU -> MaxPool
      |
      v
Conv2d -> ReLU -> MaxPool
      |
      v
Conv2d -> ReLU
      |
      v
Adaptive Average Pool
      |
      v
Linear classifier
      |
      v
10 class probabilities
```

## What I am learning

- PyTorch tensors and neural-network modules
- convolutional neural networks (CNNs)
- datasets and DataLoaders
- image augmentation and normalization
- forward propagation and loss calculation
- backpropagation and gradient descent
- Adam optimization
- train vs evaluation mode
- saving/loading model weights
- inference on a new image
- CPU/GPU device handling

## Project structure

```text
PyTorch-Image-classifier/
├── src/
│   ├── __init__.py
│   ├── data.py
│   └── model.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/madhusrimathi/PyTorch-Image-classifier.git
cd PyTorch-Image-classifier
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

CIFAR-10 downloads automatically. The default run trains for 10 epochs and saves the checkpoint to `models/cifar10_cnn.pt`.

For a quick test:

```bash
python train.py --epochs 1
```

## Predict an image

After training:

```bash
python predict.py path/to/image.jpg
```

The script resizes the image, applies the same normalization used during training, runs inference, and prints the predicted class and confidence.

## Core PyTorch concepts

### `nn.Module`
`SimpleCNN` inherits from `nn.Module`. Layers are declared in the constructor and the forward pass describes how an input tensor moves through them.

### Training loop
Each training batch follows:

```text
images -> model -> logits -> loss
                         |
                         v
                    backward()
                         |
                         v
                  optimizer.step()
```

`loss.backward()` calculates gradients. `optimizer.step()` uses those gradients to update the model parameters.

### Why CrossEntropyLoss?
This is a multi-class classification problem where every image belongs to one of ten classes. `CrossEntropyLoss` compares the model's raw logits with the correct class label.

## Next improvements

- plot training and validation curves
- add a confusion matrix and per-class metrics
- compare this CNN with transfer learning
- expose inference through a small web interface
- experiment with hyperparameters and document the results

## Note

This is a learning-focused implementation. Results depend on training settings and hardware; the repository does not claim production-level accuracy.
