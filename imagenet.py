import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time

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
        self.fc3 = nn.Linear(256, 10)
        self.fc4 = nn.Linear(10, 10)

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

data = np.load('data/training.npy')
label = np.load('data/labels.npy')[:,0, np.newaxis]

label = np.eye(10)[label.reshape(-1)]
label = label[:, np.newaxis, :]
data = data[:, np.newaxis, :, :]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1)
epochs = 100

def train():
    all_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        for i in tqdm(range(13580)):
            inputs, labels = data[i], label[i]
            inputs, labels = torch.from_numpy(inputs).float().to(device), torch.tensor(labels).float().to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        end_time = time.time()
        epoch_duration = end_time - start_time
        all_losses.append(epoch_loss / len(data))

        print(f"Epoch {epoch + 1}/{epochs}, loss: {epoch_loss / len(data)}, time: {epoch_duration:.2f}s")

train()