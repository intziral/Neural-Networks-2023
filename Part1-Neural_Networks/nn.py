import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from timm.utils import AverageMeter
import time
import os

from models.small_nn import Small_Linear
from models.big_nn import Big_Linear
from models.small_conv import Conv


if __name__ == '__main__':

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Hyperparameters
    batch_size = 4
    epoch_num = 50
    lr = 0.001

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print("Training set has " + str(len(trainset)) + " samples.")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("Testing set has " + str(len(testset)) + " samples.")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # function to show images
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    models = (Small_Linear(), Big_Linear(), Conv())

    train_accuracies = np.empty((len(models)))
    train_times = np.empty((len(models)))
    test_accuracies = np.empty((len(models)))

    # TRAINING
    for m, model in enumerate(models):

        PATH = f'./net{m+1}.pth'

        if not os.path.isfile(PATH):

            net = model
            # Training on GPU
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(device)
            net.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

            # Training the netowrk
            start_time = time.time()
            print(f"Starting Training Model {m+1} ...")

            for epoch in range(epoch_num):  # loop over the dataset multiple times

                running_loss = AverageMeter()
                progress_bar = tqdm(trainloader)
                for i, data in enumerate(progress_bar):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss.update(loss.item())
                    progress_bar.set_postfix({"last_loss": loss.detach().item(), "epoch_loss": running_loss.avg})

                    if i % 2000 == 1999:    # print every 2000 mini-batches
                        print(f'[epoch: {epoch + 1}, {i + 1:5d}], loss: {running_loss.avg}')
                        running_loss.reset()  
            
            # Save trained model
            torch.save(net.state_dict(), PATH)
                
        # Print training accuracy
        net.eval()
        correct_train = 0
        total_train = 0

        with torch.no_grad():
            for inputs, labels in trainloader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train
        print(f'Train Accuracy: {train_acc * 100:.2f}%')

        training_time = (time.time() - start_time) / 60
        print(f"\nTraining Time: {training_time:.2f} minutes")

        train_accuracies[m] = train_acc
        train_times[m] = training_time

        # Load model
        net = model
        net.load_state_dict(torch.load(PATH))
        
        labs = torch.tensor([], dtype=int)
        preds = torch.tensor([], dtype=int)

        # TESTING
        with torch.no_grad():
            for i, data in enumerate(testloader):
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                predicted = torch.argmax(outputs, dim = -1)
                preds = torch.cat((preds, predicted))
                labs = torch.cat((labs, labels))

        results = preds == labs
        accuracy = sum(results.to(int)) / len(results)
        print(f'Accuracy of the network on the 10000 test images: {100 * accuracy} %')

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
