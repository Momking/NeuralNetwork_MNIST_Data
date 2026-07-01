import sys

import torch
from model import NeuralNetwork
from PIL import Image
from torchvision import transforms


def predict(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.Grayscale(),  # Converting to grayscale
            transforms.Resize((28, 28)),  # Resize to MNIST size
            transforms.ToTensor(),  # Converting to tensor
        ]
    )

    image = Image.open(image_path)
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    print(f"Predicted digit: {prediction}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
