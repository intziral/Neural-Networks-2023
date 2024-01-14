import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten images and normalize rgb values to [0, 1]
X_train, X_test = X_train.reshape(-1, 3072) / 255.0, X_test.reshape(-1, 3072) / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

# summarize loaded dataset
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

## KNN
neighbors = np.array([1, 3])
metrics = np.array(['euclideian', 'cosine']) # different distance metrics
knn_train_accuracy = np.empty([len(neighbors), len(metrics)])
knn_test_accuracy = np.empty([len(neighbors), len(metrics)])

# Loop over neighbors values 
for i, k in enumerate(neighbors):

    # Loop over different neighbor distance metrics
    for j, m in enumerate(metrics):

        # KNN Classifier
        knn = KNeighborsClassifier(n_neighbors = k, metric = metrics[j])
        knn.fit(X_train, y_train)

        start = time.time()  
        # Compute training and test data accuracy 
        knn_train_accuracy[i, j] = knn.score(X_train, y_train)
        knn_test_accuracy[i, j] = knn.score(X_test, y_test)
        
        # record end time
        end = time.time()
        print(f"Testing time of {k}-Nearest Neighbors was:", round((end-start) / 60.0, 2), f"mins ({m})")

## NEAREST CENTROID

# NC Classifier
start = time.time()
ncc = NearestCentroid()
ncc.fit(X_train, y_train)
end = time.time()

print("Training time of Nearest Centroid Classifier was:", round((end-start) * 10**3 / 60.0, 2), "secs")

# Compute training and test data accuracy
start = time.time()
ncc_train_accuracy = ncc.score(X_train, y_train)
ncc_test_accuracy = ncc.score(X_test, y_test)
end = time.time()

print("Testing time of Nearest Centroid Classifier was:", round((end-start) * 10**3 / 60.0, 2), "secs")

## PLOTS
# Euclideian KNN
classifiers = ("KNN-1", "KNN-3")
classifier_metrics= {
    'Training Accuracy': (knn_train_accuracy[0, 0], knn_train_accuracy[1, 0]),
    'Testing Accuracy': (knn_test_accuracy[0, 0], knn_test_accuracy[1, 0]),
}

x = np.arange(len(classifiers))  # the label locations
width = 0.3  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in classifier_metrics.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('KNN with Euclideian Distance')
ax.set_xticks(x + width / 2, classifiers)
ax.legend(loc='upper right')
ax.set_ylim(0, 1)
plt.show()

# Cosine KNN
classifiers = ("KNN-1", "KNN-3")
classifier_metrics= {
    'Training Accuracy': (knn_train_accuracy[0, 1], knn_train_accuracy[1, 1]),
    'Testing Accuracy': (knn_test_accuracy[0, 1], knn_test_accuracy[1, 1]),
}

x = np.arange(len(classifiers))  # the label locations
width = 0.3  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in classifier_metrics.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('KNN with Cosine Similarity')
ax.set_xticks(x + width / 2, classifiers)
ax.legend(loc='upper right')
ax.set_ylim(0, 1)
plt.show()

# NCC Results
print("Training accuracy of Nearest Centroid:", ncc_train_accuracy)
print("Testing accuracy of Nearest Centroid:", ncc_test_accuracy)





