################################################################################
################################ Import libraries ##############################
################################################################################
print("Loading Libraries...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pickle



################################################################################
################################## Load the data ###############################
################################################################################
print("Loading and Format Dataset...")

##### Read in dataset
data = pd.read_csv("DBHImageExtraction_185kVect977Elements-2023-01-18.csv", sep = ",")


##### Define data columns, if you have rownames you need to ignore
print("Edit Away Rownames...")


### x features
data_columns = data.columns[10:]
new_data = data[ data_columns ]


### y-vector
yvct = data.columns[0:]
new_yvct = data[ yvct ]


##### Split the data into inputs (X) and labels (y)
print("Split Data from y-vector...")
X = data.iloc[:, 10:].values
y = new_yvct.iloc[:, 0].values


##### Convert the data to PyTorch tensors
print("Convert to pytorch tensors...")
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
# y = y.to(dtype = torch.long)


##### Split the data into inputs (X) and labels (y)
print("Build test train split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


##### Create a dataset from the training data
print("Build training set...")
train_dataset = TensorDataset(X_train, y_train)


##### Convert to Torch Tensor format
y_train = torch.tensor(y_train, dtype=torch.long) #, device=device)
x_train = torch.tensor(X_train, dtype=torch.long) #, device=device)
y_test = torch.tensor(y_test, dtype=torch.long) #, device=device)
X_test = torch.tensor(X_test, dtype=torch.long) #, device=device)


################################################################################
############################## Transformer Model ###############################
################################################################################
print("Create Transformer Model...")

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Create an instance of the Transformer class
print("Create Tranformer Class...")

input_dim = X_train.shape[1]
hidden_dim = 512
output_dim = 2
model = Transformer(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
print("Define Loss function...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Train the model
print("Train Model...")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for epoch in range(100):
    for X, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    print("Epoch: ", epoch, "Loss: ", loss.item())




################################################################################
#################### Evaluate the model on the test data #######################
################################################################################
print("Evaluate Model...")

with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.argmax(dim=1)
    accuracy = (y_pred == y_test).float().mean()
    print("Test Accuracy: ", accuracy.item())
