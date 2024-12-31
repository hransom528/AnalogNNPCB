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
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import dataset
print("Loading dataset...")
heart_df = pd.read_csv("./ecg-arrhythmia-classification-dataset/MIT-BIH-Arrhythmia-Database.csv")

# Encode type column
print("Processing dataset...")
type_col = heart_df["type"].to_numpy()
type_mat = np.zeros((len(type_col), 5))
#print(np.unique(type_col))
for i in range(len(type_col)):
    item = type_col[i]
    if (item == "F"):
        type_mat[i] = np.array([0,0,0,0,1]) #1
    elif (item == "N"):
        type_mat[i] = np.array([0,0,0,1,0]) #2
    elif (item == "Q"):
        type_mat[i] = np.array([0,0,1,0,0]) #3
    elif (item == "SVEB"):
        type_mat[i] = np.array([0,1,0,0,0]) #4
    elif (item == "VEB"):
        type_mat[i] = np.array([1,0,0,0,0]) #5

# Remove unnecessary columns
heart_df = heart_df.drop("record", axis=1)
heart_df = heart_df.drop("type", axis=1)

# Encode data into tensor objects
print("Encoding data...")
TEST_SIZE = 0.1
SEED = np.random.randint(0, 4294967294, dtype=np.uint32)
heart_df_mat = heart_df.to_numpy()
data = TensorDataset(torch.tensor(heart_df_mat).type(torch.double), torch.tensor(type_mat).type(torch.double))
#stratums = np.unique(type_mat, axis=0)
train_indices, test_indices, _, _ = train_test_split(
    range(len(data)),
    type_mat,
    #stratify=type_mat,
    test_size=TEST_SIZE,
    random_state=SEED
)

# Generate subset based on indices
batch_size = 1
train_split = Subset(data, train_indices)
test_split = Subset(data, test_indices)
train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_split, batch_size=batch_size, shuffle=True)

#training_data, test_data = random_split(heart_df, [0.8, 0.2])
#training_data = torch.tensor(heart_df_mat[:90000]).type(torch.double)
#test_data = torch.tensor(heart_df_mat[90001:]).type(torch.double)
# Encode labels into tensor objects
#training_labels = torch.tensor(type_mat[:90000]).type(torch.double)
#test_labels = torch.tensor(type_mat[90001:]).type(torch.double)
# Encode data and label tensors into datasets
#train_dataset = TensorDataset(training_data, training_labels)
#test_dataset = TensorDataset(test_data, test_labels)
# Load datasets into dataloaders
#train_dataloader = DataLoader(train_dataset, batch_size=1)
#test_dataloader = DataLoader(test_dataset, batch_size=1)

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
                nn.Linear(32, 6),
                nn.Sigmoid(),
                nn.Linear(6, 5)
            )
        else:
            self.linear_stack = []
            self.linear_stack.append(nn.Linear(32, 6))
            self.linear_stack.append(nn.Sigmoid())
            for i in range(numLayers - 2):
                self.linear_stack.append(nn.Linear(6, 6))
                self.linear_stack.append(nn.Sigmoid())
            self.linear_stack.append(nn.Linear(6, 5))
            self.linear_stack = nn.Sequential(self.linear_stack)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

# Initialize the models with given hyperparameters
learning_rate = 1e-3
epochs = 5
loss_fn = nn.CrossEntropyLoss() # Initialize the loss function TODO: Determine correct loss function
models = []
for layerCount in range(2, 7):
    model = NeuralNetwork(layerCount).to(device) # Initialize neural network onto compute device
    print(model)
    models.append(model)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Initialize optimizer based on model parameters

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
        if batch % 1000 == 0:
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
            #print(torch.argmax(pred), torch.argmax(y))
            if (torch.argmax(pred) == torch.argmax(y)):
                correct += 1

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

# Train and test model
accuracies = np.zeros(5)
loss_vectors = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    for i in range(len(models)):
        losses = train_loop(train_dataloader, model, loss_fn, optimizer)
        #loss_vectors.append(losses)
        accuracy = test_loop(test_dataloader, model, loss_fn)
        accuracies[i, t] = accuracy
print("Done!")

# Plot accuracy and training loss over time
x = np.arange(epochs)+1
for i in range(len(models)):
    plt.plot(x, accuracies[i], label=f"model{i+2}")
plt.title("ECG Anomaly Detection Accuracy")
plt.ylim([0, 100])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

'''
plt.figure()
x = np.arange(len(loss_vectors[epochs-1]))
plt.plot(x, loss_vectors[epochs-1])
plt.title("Last-epoch training loss")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.show()'''