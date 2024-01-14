import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle

if __name__ == '__main__':

    batch_size = 512

    # Load pca
    with open('Part3-RBF/pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    # Load CIFAR-10 dataset and PCA transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Lambda(lambda x: x.numpy().reshape(-1, 3072)), # Flatten
        transforms.Lambda(lambda x: pca.transform(x)),   
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze()),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print("Training set has", len(train_dataset), "samples.")
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("Testing set has", len(test_dataset), "samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    max_val = torch.zeros(99)
    min_val = torch.tensor([float("inf")] * 99)

    for inputs, _ in train_loader:
        mx, _ = inputs.max(dim=0)
        mn, _ = inputs.min(dim=0)
        max_val = torch.maximum(max_val, mx)
        min_val = torch.minimum(min_val, mn)
    
    print(max_val, min_val)

    # Save pca
    with open("Part3-RBF/mx.pkl",'wb') as f:
        pickle.dump(max_val, f)

    with open("Part3-RBF/mn.pkl",'wb') as f:
        pickle.dump(min_val, f)
    