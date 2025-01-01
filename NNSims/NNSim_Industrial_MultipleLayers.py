# Harris A. Ransom
# Analog Neural Network Test Simulations
# 12/16/2024

# Imports
import torch
import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
#from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Data splitting
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Confusion matrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import dataset
print("Loading dataset...")
industrial_df = pd.read_csv("./IndustrialDataset/data.csv")
labels = industrial_df["fail"].to_numpy()
industrial_df.drop("fail", axis=1, inplace=True)

# Encode data into tensor objects
print("Encoding data...")
TEST_SIZE = 0.1
SEED = np.random.randint(0, 4294967294, dtype=np.uint32)
industrial_df_mat = industrial_df.to_numpy()
data = TensorDataset(torch.tensor(industrial_df_mat).to(torch.long), torch.tensor(labels).to(torch.long))
train_indices, test_indices, _, _ = train_test_split(
    range(len(data)),
    labels,
    stratify=labels,
    test_size=TEST_SIZE,
    random_state=SEED
)

# Generate subset based on indices
batch_size = 1
train_split = Subset(data, train_indices)
test_split = Subset(data, test_indices)
train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_split, batch_size=batch_size, shuffle=True)

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
    def __init__(self, numLayers):
        super().__init__()
        self.flatten = nn.Flatten()

        # Dynamically create layers
        if (numLayers <= 1):
            raise ValueError("Invalid number of layers!")
        
        if (numLayers == 2):
            self.linear_stack = nn.Sequential(
                nn.Linear(9, 6),
                nn.Sigmoid(),
                nn.Linear(6, 1),
                nn.Sigmoid()
            )
        else:
            self.linear_stack = []
            self.linear_stack.append(nn.Linear(9, 6))
            self.linear_stack.append(nn.Sigmoid())
            for i in range(numLayers - 2):
                self.linear_stack.append(nn.Linear(6, 6))
                self.linear_stack.append(nn.Sigmoid())
            self.linear_stack.append(nn.Linear(6, 1))
            self.linear_stack.append(nn.Sigmoid())
            self.linear_stack = nn.Sequential(*self.linear_stack)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

# Hinge Loss
def HingeLoss(pred, y):
    if (y == 0):
        y = -1
    return torch.max(torch.tensor(0), 1 - pred*y)

# Initialize the models with given hyperparameters
learning_rate = 1e-6
momentum = 0.9
epochs = 10
loss_fn = nn.BCELoss() # Initialize the loss function
models = []
optimizers = []
for layerCount in range(2, 7):
    model = NeuralNetwork(layerCount).to(device) # Initialize neural network onto compute device
    print(model)
    models.append(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Initialize optimizer based on model parameters
    optimizers.append(optimizer)

# Neural Network Training
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X.to(torch.float32)).squeeze()
        #pred = torch.round(pred)
        y = y.squeeze().to(torch.float32)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        
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
    y_test = []
    predictions = []
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(torch.float32)).squeeze()
            y = y.squeeze().to(torch.float32)
            test_loss += loss_fn(pred, y).item()

            if (torch.round(pred) == y):
                correct += 1
            
            y_test.append(y)
            predictions.append(torch.round(pred))

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, y_test, predictions

# Train and test model
accuracies = np.zeros((5, epochs))
#loss_vectors = []
y_tests = []
prediction_sets = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    for i in range(len(models)):
        print(f"Model {i+2}")
        losses = train_loop(train_dataloader, models[i], loss_fn, optimizers[i])
        #loss_vectors.append(losses)
        accuracy, y_test, predictions = test_loop(test_dataloader, models[i], loss_fn)
        accuracies[i, t] = accuracy

        # Save data for confusion matrices on last epoch
        if (t == epochs - 1):
            y_tests.append(y_test)
            prediction_sets.append(predictions)
print("Done!")

# Plot accuracy and training loss over time
x = np.arange(epochs)+1
for i in range(len(models)):
    plt.plot(x, accuracies[i], label=f"{i+2} Layers")
plt.title("Industrial Machine Anomaly Detection Accuracy")
plt.ylim([0, 100])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Generate confusion matrices
for i in range(len(models)):
    cm = confusion_matrix(y_tests[i], prediction_sets[i])
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

'''
plt.figure()
x = np.arange(len(loss_vectors[epochs-1]))
plt.plot(x, loss_vectors[epochs-1])
plt.title("Last-epoch training loss")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.show()'''