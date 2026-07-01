import torch
from model import NeuralNetwork
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
