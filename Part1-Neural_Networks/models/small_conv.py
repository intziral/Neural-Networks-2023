import torch.nn as nn
import torch
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same")
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same")
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block(3, 32)
        self.block2 = Block(32, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(32, 128)
        self.norm1 = nn.BatchNorm1d(128)
        # output layer
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.linear1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    inp = torch.zeros((10, 3, 32, 32))

    model = Conv()

    out = model(inp)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)