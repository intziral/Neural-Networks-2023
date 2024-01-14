import torch
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_data, labels = next(iter(train_loader))
    print("Data shape:", train_data.shape)

    train_data = train_data.numpy().reshape(-1, 3072)
    print('Data reshaped:', train_data.shape)

    PATH = 'Part3-RBF/pca.pkl'

    if os.path.isfile(PATH):
        with open(PATH, 'rb') as f:
            pca = pickle.load(f)

    else:
        # Apply PCA
        pca = PCA(n_components=0.90)
        pca.fit(train_data)

        # Save pca
        with open(PATH,'wb') as f:
                pickle.dump(pca, f)
        print('Saved PCA')
    
    print('PCA feature length:', len(pca.explained_variance_ratio_))
    
    example = train_data[1]
    components = pca.transform(example[np.newaxis, :])
    reduced_im = pca.inverse_transform(components).reshape(3, 32, 32)
    plt.imshow(np.transpose(reduced_im, (1, 2, 0)))
    plt.title(classes[labels[1]])
    plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import cifar10
# from sklearn import svm
# from sklearn.decomposition import PCA
# import os
# import pickle

# # load dataset
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # flatten images and normalize rgb values to [0, 1]
# X_train = X_train.reshape(-1, 3072) / 255.0

# # flatten the label values
# y_train = y_train.flatten()

# PATH = 'PART 3 - RBF/pca.pkl'

# if os.path.isfile():
#     with open(PATH, 'rb') as f:
#         X_pca = pickle.load(f)

# else:
#     # PCA
#     X_pca = PCA(n_components=0.90)
#     X_pca.fit(X_train)
#     print(X_pca.explained_variance_ratio_)
#     # save pca
#     with open(PATH,'wb') as f:
#             pickle.dump(X_pca, f)

# # show example images before and after pca
# example = X_train[1]
# components = X_pca.transform(example[np.newaxis, :])
# reduced_im = X_pca.inverse_transform(components)
# print("Feature size after PCA: ", components.shape) 

# fig, axes = plt.subplots(2, 1)
# axes[0].imshow(X_train[1].reshape(32, 32, 3))
# axes[1].imshow(reduced_im.reshape(32, 32, 3))
# plt.show()
    

    
    

