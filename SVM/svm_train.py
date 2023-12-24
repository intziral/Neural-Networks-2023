import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn import svm
from sklearn.decomposition import PCA
import os
import pickle

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten images and normalize rgb values to [0, 1]
X_train, X_test = X_train.reshape(-1, 3072) / 255.0, X_test.reshape(-1, 3072) / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

# summarize loaded dataset
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

if os.path.isfile('SVM/models/train_pca.pkl'):
    with open('SVM/models/train_pca.pkl', 'rb') as f:
        X_pca = pickle.load(f)

else:
    # PCA
    X_pca = PCA(n_components=0.90)
    X_pca.fit(X_train)
    print(X_pca.explained_variance_ratio_)
    # save pca
    with open('SVM/models/train_pca.pkl','wb') as f:
            pickle.dump(X_pca, f)

# show example images before and after pca
example = X_train[1]
components = X_pca.transform(example[np.newaxis, :])
reduced_im = X_pca.inverse_transform(components)
print("Feature size after PCA: ", components.shape) 

fig, axes = plt.subplots(2, 1)
axes[0].imshow(X_train[1].reshape(32, 32, 3))
axes[1].imshow(reduced_im.reshape(32, 32, 3))
plt.show()

# transform input
X_train_pca = X_pca.transform(X_train)

C = 1.0  # SVM regularization parameter
models = [
    svm.LinearSVC(C=C, dual="auto"),
    svm.SVC(kernel="linear", C=C),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
]

# train models
models = [clf.fit(X_train_pca, y_train) for clf in models]

# save models
for i, model in enumerate(models):
    with open(f'SVM/models/model{i+1}.pkl','wb') as f:
        pickle.dump(model, f)


