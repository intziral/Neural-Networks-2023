import torch.nn as nn
import torch.nn.functional as F

class Big_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3072, 1024)
        # hidden layers
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        # output layer
        self.linear4 = nn.Linear(128, 10)

    def forward(self, x):
        # flatten images into vectors
        x = x.view(x.size(0), -1)
        # layers and activation functions
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        return x