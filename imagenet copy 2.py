import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = dataset.data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8*163*163, 1024)
        self.fc2 = nn.Linear(1024, 13)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=12, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        # self.conv2 = nn.Conv2d(8, 12, kernel_size=5, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=0)
        # self.fc1 = nn.Linear(16 * 17 * 17, 1024)
        # self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(256, 10)
        # self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x) 
        x = x.view(-1, 8*163*163)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# data = np.load('data/training.npy')
# label = np.load('data/labels.npy')[:,0, np.newaxis]

# label = np.eye(10)[label.reshape(-1)]
# label = label[:, np.newaxis, :]
# data = data[:, np.newaxis, :, :]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
epochs = 10000

def train():
    all_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        all_labels = []
        all_preds = []
        for i in tqdm(range(100)):
            inputs, labels = data[i]
            inputs, labels = torch.from_numpy(inputs).float().to(device), torch.tensor(labels).float().to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.detach().cpu().numpy())

        end_time = time.time()
        epoch_duration = end_time - start_time
        all_losses.append(epoch_loss / len(data))

        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_preds = np.argmax(all_preds, axis=1)
        all_labels = np.argmax(all_labels, axis=1)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1}/{epochs}, loss: {epoch_loss / len(data)}, time: {epoch_duration:.2f}s, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
        
        # Initialize metrics list on the first epoch
        if epoch == 0:
            metrics = []

        # Append current epoch metrics
        metrics.append({
            'Epoch': epoch + 1,
            'Loss': epoch_loss / len(data),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })

        # Every 10 epochs, save metrics and model
        if (epoch + 1) % 10 == 0:
            df_metrics = pd.DataFrame(metrics)
            df_metrics.to_csv(f"toy_indicators/metrics_epoch_{epoch+1}.csv", index=False)
            torch.save(net.state_dict(), f"toy_models/model_epoch_{epoch+1}.pth")

train()