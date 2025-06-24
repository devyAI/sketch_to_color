import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataset():
    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load CIFAR-10 dataset
    try:
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Attempting to download manually...")
        import os
        os.makedirs('./data', exist_ok=True)
        
        # Download the dataset manually
        from torchvision.datasets.utils import download_and_extract_archive
        urls = [
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        ]
        
        for url in urls:
            download_and_extract_archive(
                url,
                download_root='./data',
                extract_root='./data'
            )
        
        # Load the dataset again
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=train_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=False,
            transform=test_transform
        )
    
    return train_dataset, test_dataset

def get_data_loaders(batch_size=32):
    train_dataset, test_dataset = get_cifar10_dataset()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

def convert_to_grayscale(image):
    """Convert RGB image to grayscale"""
    return torch.mean(image, dim=0, keepdim=True)

def create_grayscale_dataset(dataset):
    """Create a dataset with grayscale images"""
    grayscale_dataset = []
    for image, label in dataset:
        grayscale_image = convert_to_grayscale(image)
        grayscale_dataset.append((grayscale_image, image))
    return grayscale_dataset
