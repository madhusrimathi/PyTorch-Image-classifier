from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(data_dir="data", batch_size=64, num_workers=2):
    root = Path(data_dir)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, train_set.classes
