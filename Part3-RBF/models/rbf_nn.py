import torch_rbf as rbf
import torch.nn as nn
import sys

class Network(nn.Module):
    
    def __init__(self, in_features, num_centers, num_classes, basis_func):
        super(Network, self).__init__()
        self.rbf_layer = rbf.RBF(in_features, num_centers, basis_func)
        self.linear_layer = nn.Linear(num_centers, num_classes)
    
    def forward(self, x):
        x = self.rbf_layer(x)
        x = self.linear_layer(x)
        return x
    
    def fit(self, train_loader, epochs, optimiser, loss_func):
        self.train()
        epoch = 0
        while epoch < epochs:
            epoch += 1
            current_loss = 0
            batches = 0
            progress = 0
            for inputs, labels in train_loader:
                batches += 1
                optimiser.zero_grad()
                outputs = self.forward(inputs)
                loss = loss_func(outputs, labels)
                current_loss += loss.item()
                loss.backward()
                optimiser.step()
                progress += labels.size(0)
                sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f      ' % \
                                 (epoch, progress, 50000, current_loss/batches))