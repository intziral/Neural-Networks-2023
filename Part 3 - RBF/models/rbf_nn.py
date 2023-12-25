import torch.nn as nn
import torch
import torch.nn.functional as F

# Define the RBF layer
class RBFLayer(nn.Module):
    def __init__(self, in_features, num_centers, gamma=1.0):
        super(RBFLayer, self).__init__()
        self.centers = nn.Parameter(torch.Tensor(num_centers, in_features))
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        nn.init.xavier_uniform_(self.centers.data)

    def forward(self, x):
        x = x.unsqueeze(1) - self.centers.unsqueeze(0)
        x = torch.norm(x, dim=-1)
        x = torch.exp(-self.gamma * x)
        return x

# Define the RBFNN model
class RBFNN(nn.Module):
    def __init__(self, in_features, num_centers, num_classes):
        super(RBFNN, self).__init__()
        self.rbf = RBFLayer(in_features, num_centers)
        self.linear = nn.Linear(num_centers, num_classes)

    def forward(self, x):
        x = self.rbf(x)
        x = self.linear(x)
        return x