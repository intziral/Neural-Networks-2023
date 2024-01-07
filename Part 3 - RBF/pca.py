import torch
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import numpy as np
import os
import pickle

if __name__ == '__main__':

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50000, shuffle=True, num_workers=2)

    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    image_data = images.numpy().reshape(-1, 3072)

    print('Images have shape:', image_data.shape)

    PATH = 'PART 3 - RBF/pca.pkl'
    pca_components = 0.9

    if os.path.isfile(PATH):
        with open(PATH, 'rb') as f:
            pca = pickle.load(f)

    else:
        # Apply PCA
        pca = PCA(n_components=pca_components)
        pca.fit(image_data)

        # Save pca
        with open(PATH,'wb') as f:
                pickle.dump(pca, f)
        print('Saved PCA')
    
    print('PCA feature length:', len(pca.explained_variance_ratio_))
    

    
    

