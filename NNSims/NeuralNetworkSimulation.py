# Harris A. Ransom
# Analog Neural Network Test Simulations
# 12/16/2024

# Imports
import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import dataset
heart_df = pd.read_csv("./ecg-arrhythmia-classification-dataset/MIT-BIH-Arrhythmia-Database.csv")

# Encode type column
type_col = heart_df["type"].to_numpy()
type_mat = np.zeros((len(type_col), 5))
#print(np.unique(type_col))
for i in range(len(type_col)):
    item = type_col[i]
    if (item == "F"):
        type_mat[i] = np.array([0,0,0,0,1])
    elif (item == "N"):
        type_mat[i] = np.array([0,0,0,1,0])
    elif (item == "Q"):
        type_mat[i] = np.array([0,0,1,0,0])
    elif (item == "SVEB"):
        type_mat[i] = np.array([0,1,0,0,0])
    elif (item == "VEB"):
        type_mat[i] = np.array([1,0,0,0,0])

# Remove unnecessary columns
heart_df = heart_df.drop("record", axis=1)
heart_df = heart_df.drop("type", axis=1)

# Encode data into tensor objects
heart_df_mat = heart_df.to_numpy()
training_data = torch.tensor(heart_df_mat[:90000]).type(torch.double)
test_data = torch.tensor(heart_df_mat[90001:]).type(torch.double)

# Encode labels into tensor objects
training_labels = torch.tensor(type_mat[:90000]).type(torch.double)
test_labels = torch.tensor(type_mat[90001:]).type(torch.double)

# Encode data and label tensors into datasets
train_dataset = TensorDataset(training_data, training_labels)
test_dataset = TensorDataset(test_data, test_labels)

# Load datasets into dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# Determine compute device being used
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define neural network
# See: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(32, 6),
            nn.Sigmoid(), # Sigmoid is used to approximate the nonlinear diode activation function
            nn.Linear(6, 6),
            nn.Sigmoid(),
            nn.Linear(6, 6),
            nn.Sigmoid(),
            nn.Linear(6, 6),
            nn.Sigmoid(),
            nn.Linear(6, 6),
            nn.Sigmoid(),
            nn.Linear(6, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

learning_rate = 1e-3
batch_size = 64
epochs = 5
loss_fn = nn.CrossEntropyLoss() # Initialize the loss function TODO: Determine correct loss function
model = NeuralNetwork().to(device) # Initialize neural network onto compute device
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Initialize optimizer based on model parameters
print(model)

# Neural Network Training
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(torch.float32))
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses = []
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            losses.append(loss)
    return losses

# Neural Network Testing
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(torch.float32))
            test_loss += loss_fn(pred, y).item()
            if (torch.argmax(pred) == torch.argmax(y)):
                correct += 1

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

# Train and test model
epochs = 5
accuracies = []
loss_vectors = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    losses = train_loop(train_dataloader, model, loss_fn, optimizer)
    loss_vectors.append(losses)
    accuracy = test_loop(test_dataloader, model, loss_fn)
    accuracies.append(accuracy)
print("Done!")

# Plot accuracy and training loss over time
x = np.arange(epochs)+1
plt.plot(x, accuracies)
plt.title("ECG Anomaly Detection Accuracy")
plt.ylim([0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.figure()
x = np.arange(len(losses[epochs-1]))
plt.plot(x, loss_vectors[epochs-1])
plt.title("Last-epoch training loss")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.show()