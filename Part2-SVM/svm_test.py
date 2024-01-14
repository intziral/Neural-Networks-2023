import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import os
import pickle

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
labels = ["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# flatten images and normalize rgb values to [0, 1]
X_train, X_test = X_train.reshape(-1, 3072) / 255.0, X_test.reshape(-1, 3072) / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

# load pca
with open('SVM/models/train_pca.pkl', 'rb') as f:
    X_pca = pickle.load(f)

# transform train and test sets
X_test_pca = X_pca.transform(X_test)
X_train_pca = X_pca.transform(X_train)
print("Train set size after PCA:", X_train_pca.shape)
print("Test set size after PCA:", X_test_pca.shape)

# init arrays
models = ["Linear SVC", "Linear Kernel", "RBF 0.7", "RBF auto", "Polynomial"]
model_train_accuracy = np.empty([len(models)])
model_test_accuracy = np.empty([len(models)])

# load and test models
for i in range(len(models)):

    # open trained model
    with open(f'SVM/models/model{i+1}.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    # classification
    y_pred_train = clf.predict(X_train_pca)
    y_pred = clf.predict(X_test_pca)

    # results
    train_results = y_pred_train == y_train # boolean correct or not classification
    test_results = y_pred == y_test

    # calculate accuracy
    model_train_accuracy[i] = sum(train_results) / len(train_results)
    model_test_accuracy[i] = sum(test_results) / len(test_results)
    print("Training accuracy of model", i+1, "=", round(100 * model_train_accuracy[i], 3), "%")
    print("Accuracy of model", i+1, "on the testing set =",  round(100 * model_test_accuracy[i], 3), "%")

    # categorize predictions
    correct_pred = np.where(test_results)[0]
    incorrect_pred = np.where(test_results == False)[0]

    # Classification examples
    image_no = 5 # show 5 correct testing classification examples

    if not os.path.isfile(f"SVM/figures/model{i+1}_correct.png"):
        
        fig, axes = plt.subplots(2, image_no)
        for k in range(image_no):
            j = np.random.choice(correct_pred, size=1, replace=False)
            j = int(j)
            reduced_im = X_pca.inverse_transform(X_test_pca[j])

            # show original image
            axes[0, k].imshow(X_test[j].reshape(32, 32, 3))
            axes[0, k].set_title(f"True: {labels[y_test[j]]}")
            axes[0, k].axis('off')

            # show reduced image
            axes[1, k].imshow(reduced_im.reshape(32, 32, 3))
            axes[1, k].set_title(f"Pred: {labels[y_pred[j]]}")
            axes[1, k].axis('off')

        plt.suptitle(models[i])
        plt.tight_layout()
        plt.savefig(f"SVM/figures/model{i+1}_correct.png")

    # show incorrect classification examples
    if not os.path.isfile(f"SVM/figures/model{i+1}_incorrect.png"):

        fig, axes = plt.subplots(2, image_no)
        for k in range(image_no):
            j = np.random.choice(incorrect_pred, size=1, replace=False)
            j = int(j)
            reduced_im = X_pca.inverse_transform(X_test_pca[j])

            # show original image
            axes[0, k].imshow(X_test[j].reshape(32, 32, 3))
            axes[0, k].set_title(f"True: {labels[y_test[j]]}")
            axes[0, k].axis('off')

            # show reduced image
            axes[1, k].imshow(reduced_im.reshape(32, 32, 3))
            axes[1, k].set_title(f"Pred: {labels[y_pred[j]]}")
            axes[1, k].axis('off')

        plt.suptitle(models[i])
        plt.tight_layout()
        plt.savefig(f"SVM/figures/model{i+1}_incorrect.png")

# Accuracy plots
if not os.path.isfile(f"SVM/figures/accuracies.png"):
    
    plt.figure()

    # Set up the bar positions
    bar_width = 0.35
    index = np.arange(len(models))

    # Create bar plots for training and testing accuracies
    plt.bar(index, model_train_accuracy, width=bar_width, label='Training Accuracy')
    plt.bar(index + bar_width, model_test_accuracy, width=bar_width, label='Testing Accuracy')

    # Customize the plot
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracies for Different SVCs')
    plt.xticks(index + bar_width / 2, models)
    plt.legend()

    # Save plot
    plt.savefig(f"SVM/figures/accuracies.png")

    # Display the plot
    plt.show()








