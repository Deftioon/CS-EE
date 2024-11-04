import network
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data = dataset.data

net = network.Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)
epochs = 100

def train():
    all_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        for i in range(100):
            inputs, labels = data[i]
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

if __name__ == "__main__":
    train()