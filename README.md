# Neural Network for MNIST Digit Classification

A simple handwritten digit classifier built with **PyTorch** using a fully connected neural network trained on the **MNIST** dataset.

## Features

* Train a neural network on the MNIST dataset
* Evaluate the model on the MNIST test set
* Predict digits from custom images
* Automatic dataset download
* GPU acceleration (CUDA) when available

---

## Project Structure

```text
.
├── model.py          # Neural network architecture
├── train.py          # Train the model
├── evaluate.py       # Evaluate test accuracy
├── predict.py        # Predict a digit from an image
├── .gitignore
└── README.md
```

> The trained model (`mnist_model.pth`) and dataset (`data/`) are intentionally excluded from the repository.

---

## Requirements

* Python 3.10+
* PyTorch
* TorchVision
* Pillow

Install the required packages:

```bash
pip install torch torchvision pillow
```

---

## Model Architecture

```
Input (28 × 28)
      │
      ▼
Flatten
      │
      ▼
Linear(784 → 128)
      │
      ▼
ReLU
      │
      ▼
Linear(128 → 64)
      │
      ▼
ReLU
      │
      ▼
Linear(64 → 10)
```

The output consists of 10 logits corresponding to the digits **0–9**.

---

## Training

Train the model:

```bash
python train.py
```

The MNIST dataset will be downloaded automatically if it is not already available.

After training, the model weights are saved as:

```text
mnist_model.pth
```

---

## Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

This prints the classification accuracy on the MNIST test dataset.

---

## Predicting a Digit

Predict a digit from an image:

```bash
python predict.py path/to/image.png
```

Example:

```bash
python predict.py images/1.png
```

The script automatically:

* Converts the image to grayscale
* Inverts colors (to match MNIST)
* Crops empty borders
* Resizes the digit
* Centers it on a 28×28 canvas
* Runs inference using the trained model

Example output:

```text
Prediction : 7
Confidence : 99.84%
```

---

## Dataset

This project uses the **MNIST** handwritten digit dataset.

* 60,000 training images
* 10,000 testing images
* Image size: 28×28 pixels
* Classes: digits 0–9

---

## Future Improvements

* Convolutional Neural Network (CNN)
* Batch Normalization
* Dropout
* Learning rate scheduling
* Training metrics and loss visualization
* Support for additional handwritten digit datasets

---

## License

This project is provided for educational purposes.
