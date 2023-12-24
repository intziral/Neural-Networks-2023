import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3072, 512)
        # hidden layers
        self.linear2 = nn.Linear(512, 128)
        # output layer
        self.linear3 = nn.Linear(128, 10)

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
        return x
    


if __name__ == "__main__":
    inp = torch.zeros((10, 3, 32, 32))

    model = Net()

    out = model(inp)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)