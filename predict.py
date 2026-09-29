import argparse

import torch
from PIL import Image
from torchvision import transforms

from src.data import MEAN, STD
from src.model import SimpleCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default="models/cifar10_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    classes = checkpoint["classes"]

    model = SimpleCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)[0]
        confidence, index = probabilities.max(0)

    print(f"Prediction: {classes[index.item()]}")
    print(f"Confidence: {confidence.item():.2%}")


if __name__ == "__main__":
    main()
