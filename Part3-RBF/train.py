import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import numpy as np
import time
from models.rbf_nn import Network
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pickle
import torch_rbf as rbf
import os

def main():

    # # Set random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)

    # Hyperparameters
    lr = 0.001
    num_epochs = 10
    batch_size = 16
    num_centers = (10, 20)
    # num_centers = (10, 20, 40, 80)
    basis_func = rbf.gaussian

    # Load pca and stats
    with open('Part3-RBF/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('Part3-RBF/mx.pkl', 'rb') as f:
        mx = pickle.load(f)
    with open('Part3-RBF/mn.pkl', 'rb') as f:
        mn = pickle.load(f)
    
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

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # show example images before and after pca
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    fig, axes = plt.subplots(1, batch_size)
    for i in range(batch_size):
        image_data = images[i].numpy()
        reduced_im = pca.inverse_transform(image_data).reshape(3, 32, 32)
        axes[i].imshow(np.transpose(reduced_im, (1, 2, 0)))
        axes[i].axis('off')
        axes[i].set_title(classes[labels[i]])
    plt.suptitle('Example PCA Images')
    plt.tight_layout()
    plt.show()
    
    num_features = images.numpy().shape[1]
    num_classes = len(train_dataset.classes)
    print('\nPCA images feature length:', num_features)

    model_accuracies = np.empty((num_centers))
    train_times = np.empty((num_centers))

    # Training models
    for k, c_num in enumerate(num_centers):

        PATH = f'Part3-RBF/rbf_net{k+1}.pth'

        # Initialize model
        model = Network(num_features, c_num, num_classes, basis_func)

        # Initialize KMeans for center selection
        kmeans = KMeans(n_clusters=c_num, random_state=42)

        # Get a batch of data to determine the shape
        sample_batch, _ = next(iter(train_loader))
        sample_batch = sample_batch.view(sample_batch.size(0), -1)

        # Fit KMeans to determine the centers
        kmeans.fit(sample_batch.numpy())
        centers = kmeans.cluster_centers_

        # Set the RBF layer centers to KMeans centers
        model.rbf_layer.centers.data = torch.Tensor(centers)

        # Choose optimiser and loss function
        optimiser = torch.optim.SGD(model.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        if not os.path.isfile(PATH):

            start_time = time.time()
            print(f"\nTraining Model {k+1} ...")

            # Train model
            model.fit(train_loader, num_epochs, optimiser, loss_func)
            model.eval()

            training_time = (time.time() - start_time) / 60
            print(f"\nTraining Time: {training_time:.2f} minutes")

            # save model training time 
            train_times[k] = training_time

            # Save trained model
            torch.save(model.state_dict(), PATH)

        # Load model 
        model.load_state_dict(torch.load(PATH))

        # Test the model
        labs = torch.tensor([], dtype=int)
        preds = torch.tensor([], dtype=int)

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images)
                predicted = torch.argmax(outputs, dim = -1)
                preds = torch.cat((preds, predicted))
                labs = torch.cat((labs, labels))

        results = preds == labs
        accuracy = sum(results.to(int)) / len(results)
        print(f'Accuracy of the network on the 10000 test images: {100 * accuracy} %')

        # save accuracy
        model_accuracies[k] = accuracy
    
    # Accuracy plots
    ACCS = "SVM/figures/accuracies.png"

    if not os.path.isfile(ACCS):
        
        fig, ax = plt.subplots()

        models = []
        for i, _ in enumerate(num_centers):
            models.append(f"Model {i+1}")
        
        ax.bar(models, model_accuracies)

        ax.set_ylabel('Accuracy')
        ax.set_title('Test Accuracies for Different #centers')
        ax.legend(title='Models')

        # Save plot
        plt.savefig(ACCS)

    # Display the plot
    ld_acc_plot = imread(ACCS)
    plt.imshow(ld_acc_plot)
    plt.show()
    

    # OPTIMAL MODEL
    opt_index = np.argmax(model_accuracies)
    opt_num_centers = num_centers[opt_index]
    print(model_accuracies)
    print(train_times)
    print(opt_num_centers)

    # Initialize optimal RBFNN model
    opt_model = Network(num_features, opt_num_centers, num_classes, basis_func)

    opt_epochs = 1
    PATH = f'Part3-RBF/opt_net.pth'

    if not os.path.isfile(PATH):

        start_time = time.time()
        print(f"\nTraining Model {k+1} ...")

        # Train model
        opt_model.fit(train_loader, opt_epochs, optimiser, loss_func)
        opt_model.eval()

        training_time = (time.time() - start_time) / 60
        print(f"\nTraining Time: {training_time:.2f} minutes")

        # Save trained model
        torch.save(opt_model.state_dict(), PATH)

        #         # print statistics
        #         running_loss.update(loss.item())
        #         progress_bar.set_postfix({"last_loss": loss.detach().item(), "epoch_loss": running_loss.avg})

        #         if i % 2000 == 1999:    # print every 2000 mini-batches
        #             print(f'[epoch: {epoch + 1}, {i + 1:5d}], loss: {running_loss / 2000:.3f}')
        #             running_loss = 0.0

        # Print training accuracy
        model.eval()
        correct_train = 0
        total_train = 0

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        opt_train_acc = correct_train / total_train
        print(f'Train Accuracy: {opt_train_acc * 100:.2f}%')

    # Load model
    opt_model.load_state_dict(torch.load(PATH))

    # Test the model
    opt_model.eval()
    correct_test = 0
    total_test = 0
    examples_correct = []
    examples_incorrect = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1)
            outputs = opt_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # Collect examples of correct and incorrect classifications
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    examples_correct.append((inputs[i], labels[i].item(), predicted[i].item()))
                else:
                    examples_incorrect.append((inputs[i], labels[i].item(), predicted[i].item()))

    opt_test_acc = correct_test / total_test
    print(f'Test Accuracy: {opt_test_acc * 100:.2f}%')

    # Display examples of correct and incorrect classifications
    print("\nExamples of Correct Classifications:")
    for example in examples_correct[:5]:
        print(f"Actual Class: {example[1]}, Predicted Class: {example[2]}")

    print("\nExamples of Incorrect Classifications:")
    for example in examples_incorrect[:5]:
        print(f"Actual Class: {example[1]}, Predicted Class: {example[2]}")
    
    # Accuracy plots
    OPT = "Part3-RBF/figures/opt_acc.png"

    if not os.path.isfile(OPT):
        
        fig, ax = plt.subplots()

        accuracies = [opt_train_acc, opt_test_acc]
        acc_labels = ["Training Accuracy, Testing Accuracy"]
        ax.bar(accuracies, acc_labels)

        ax.set_ylabel('Accuracy')
        ax.set_title(f'Optimal Model, #centers={opt_num_centers}')

        # Save plot
        plt.savefig(OPT)

    # Display the plot
    ld_opt_plot = imread(OPT)
    plt.imshow(ld_opt_plot)
    plt.show()

if __name__ == '__main__':
    main()


