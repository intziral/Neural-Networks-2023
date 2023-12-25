import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.cluster import KMeans
import time
from models.rbf_nn import RBFNN

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Hyperparameters
num_centers = 100
num_classes = 10
lr = 0.01
num_epochs = 10

# Initialize the RBFNN model
model = RBFNN(in_features=32*32*3, num_centers=num_centers, num_classes=num_classes)

# Training on GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Initialize KMeans for center selection
kmeans = KMeans(n_clusters=num_centers, random_state=42)

# Get a batch of data to determine the shape
sample_batch, _ = next(iter(train_loader))
sample_batch = sample_batch.view(sample_batch.size(0), -1)

# Fit KMeans to determine the centers
kmeans.fit(sample_batch.numpy())
centers = kmeans.cluster_centers_

# Set the RBF layer centers to KMeans centers
model.rbf.centers.data = torch.Tensor(centers)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training the model
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

    accuracy_train = correct_train / total_train
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Training Accuracy: {accuracy_train * 100:.2f}%')

training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Save trained model
PATH = './rbf_net.pth'
torch.save(model.state_dict(), PATH)

# Load model
model = RBFNN()
model.load_state_dict(torch.load(PATH))

# Test the model
model.eval()
correct_test = 0
total_test = 0
examples_correct = []
examples_incorrect = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        # Collect examples of correct and incorrect classifications
        for i in range(len(labels)):
            if predicted[i] == labels[i]:
                examples_correct.append((inputs[i], labels[i].item(), predicted[i].item()))
            else:
                examples_incorrect.append((inputs[i], labels[i].item(), predicted[i].item()))

accuracy_test = correct_test / total_test
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

# Display examples of correct and incorrect classifications
print("\nExamples of Correct Classifications:")
for example in examples_correct[:5]:
    print(f"Actual Class: {example[1]}, Predicted Class: {example[2]}")

print("\nExamples of Incorrect Classifications:")
for example in examples_incorrect[:5]:
    print(f"Actual Class: {example[1]}, Predicted Class: {example[2]}")


