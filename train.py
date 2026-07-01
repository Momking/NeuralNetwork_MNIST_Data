import torch  # torch → tensors
import torch.nn as nn  # nn → neural network layers
import torch.optim as optim  # optim → optimizers
from model import NeuralNetwork
from torch.utils.data import DataLoader

# torchvision → image datasets and transforms
from torchvision import datasets, transforms


def train():
    # For images we usually transform them into tensors.
    transform = transforms.ToTensor()

    # Load MNIST:
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )

    # Instead of loading all images at once, we use mini-batches.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    torch.save(model.state_dict(), "mnist_model.pth")


if __name__ == "__main__":
    train()
