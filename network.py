import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=12, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 17 * 17, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 13)
        self.fc4 = nn.Linear(13, 13)

    def forward(self, x):
        convolutional_layers = [
            (self.conv1, F.relu, self.pool),
            (self.conv2, F.relu, self.pool),
            (self.conv3, F.relu, self.pool),
        ]
        fully_connected_layers = [
            (self.fc1, F.relu, None),
            (self.fc2, F.relu, None),
            (self.fc3, F.relu, None),
            (self.fc4, F.softmax, None)
        ]
        
        for layer, activation, pool in convolutional_layers:
            x = layer(x)
            x = activation(x)
            if pool:
                x = pool(x)
        
        x = x.view(-1, 16 * 17 * 17)

        for layer, activation, pool in fully_connected_layers:
            x = layer(x)
            if activation:
                x = activation(x)
            if pool:
                x = pool(x)
        return x

if __name__ == "__main__":
    net = Net()
    test_input = torch.randn(1, 1, 500, 500)
    output = net(test_input)
    print("Output shape:", output.shape)
    print("Output:", output)