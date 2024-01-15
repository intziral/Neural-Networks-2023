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
    batch_size = 16
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
    def im_show(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show example images
    im_show(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    models = (Small_Linear(), Big_Linear(), Conv())

    train_accuracies = np.empty((len(models)))
    train_times = np.empty((len(models)))
    test_accuracies = np.empty((len(models)))

    # TRAINING
    for m, model in enumerate(models):

        PATH = f'Part1-Neural_Networks/net{m+1}.pth'

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

            losses = []
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

                losses.append(running_loss.avg)
            
            training_time = (time.time() - start_time) / 60
            print(f"\nTraining Time: {training_time:.2f} minutes")
            
            plt.plot(losses)
            plt.title(f'Model {m+1} Learning Curve')
            plt.savefig(f'Part1-Neural_Networks/figures/lr_model{m+1}.png')
            plt.show()
            
            # Save trained model
            torch.save(net.state_dict(), PATH)
        
        else:
            # Load model
            net = model.__class__()
            net.load_state_dict(torch.load(PATH))
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(device)
            net.to(device)

        # Print training accuracy
        net.eval()
        correct_train = 0
        total_train = 0

        with torch.no_grad():
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train
        print(f'Train Accuracy: {train_acc * 100:.2f}%')

        train_accuracies[m] = train_acc
        # train_times[m] = training_time
        
        labs = torch.tensor([], dtype=int).to(device)
        preds = torch.tensor([], dtype=int).to(device)
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        correct_images = []
        correct_labels = []
        correct_predictions = []
        incorrect_images = []
        incorrect_labels = []
        incorrect_predictions = []

        # TESTING
        with torch.no_grad():
            for i, data in enumerate(testloader):
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                predicted = torch.argmax(outputs, dim = -1)
                preds = torch.cat((preds, predicted))
                labs = torch.cat((labs, labels))
                # count the correct predictions for each class
                for label, prediction in zip( labels, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
                # collect the correct and incorrect predictions for each class
                for img, label, prediction in zip(images, labels, predicted):
                    if label == prediction:
                        correct_images.append(img)
                        correct_labels.append(label.item())
                        correct_predictions.append(prediction.item())
                    else:
                        incorrect_images.append(img)
                        incorrect_labels.append(label.item())
                        incorrect_predictions.append(prediction.item())

        # print total accuracy of network
        results = preds == labs
        accuracy = sum(results.to(int)) / len(results)
        print(f'Accuracy of the network on the 10000 test images: {100 * accuracy:.2f} %')

        test_accuracies[m] = accuracy

        # Print accuracy for each class
        class_accuracies = {classname: 0.0 for classname in classes}

        for classname, correct_count in correct_pred.items():
            total_count = total_pred[classname]
            if total_count > 0:
                class_accuracy = correct_count / total_count
                class_accuracies[classname] = class_accuracy
                print(f'Accuracy for class {classname:5s}: {100 * class_accuracy:.2f}%')
        
        # Visualize correct predictions
        num_correct_to_show = min(5, len(correct_images))
        for i in range(num_correct_to_show):
            plt.subplot(1, num_correct_to_show, i + 1)
            img = correct_images[i].cpu() / 2 + 0.5 # unnormalize
            npimg = np.transpose(img.numpy(), (1, 2, 0))
            plt.imshow(npimg)
            plt.title(f'True: {classes[correct_labels[i]]}, Predicted: {classes[correct_predictions[i]]}')
            plt.axis("off")
    
        plt.savefig(f'Part1-Neural_Networks/figures/correct{m+1}.png')
        plt.show()

        # Visualize incorrect predictions
        num_incorrect_to_show = min(5, len(incorrect_images))
        for i in range(num_incorrect_to_show):
            plt.subplot(1, num_incorrect_to_show, i + 1)
            img = incorrect_images[i].cpu() / 2 + 0.5 # unnormalize
            npimg = np.transpose(img.numpy(), (1, 2, 0))
            plt.imshow(npimg)
            plt.title(f'True: {classes[incorrect_labels[i]]}, Predicted: {classes[incorrect_predictions[i]]}')
            plt.axis("off")

        plt.savefig(f'Part1-Neural_Networks/figures/incorrect{m+1}.png')
        plt.show()


    ## Accuracy PLOT
    model_names = ("Small Linear", "Big Linear", "Convolutional")
    classifier_metrics= {
        'Training Accuracy': (train_accuracies[0], train_accuracies[1], train_accuracies[2]),
        'Testing Accuracy': (test_accuracies[0], test_accuracies[1], test_accuracies[2]),
    }

    x = np.arange(len(model_names))  # the label locations
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
    ax.set_title('Training and Testing Accuracies for Different Networks')
    ax.set_xticks(x + width / 2, model_names)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    plt.savefig('Part1-Neural_Networks/figures/accs.png')
    plt.show()

