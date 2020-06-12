#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2
import numpy as np
from tqdm import tqdm

# Creating a convolutional neural network (mainly for image processing)
# However, it has shown to be quicker using 1 dimensional layers (?) sequentially.
# This first block of code creates our set of training data in "training_data.npy"

# It requires folders "PetImages/Cat" & "PetImages/Dog" in the same directory to make the file.
# (Uses the Kaggle cat dog dataset)

REBUILD_DATA = False # Change this to true if you want to remake the npy training file.
# Flag to avoid building data each time.
# Preprocessing step takes a long time (with bigger data)

# It can be convenient to make a class for this, but you do not have to.

class DogsvsCats():
    IMG_SIZE = 50 # The constant size of the image we will make into.
    
    # The directories each set of pictures lies in.
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    
    # Labels, Cats = 0, Dogs = 1
    LABELS = {CATS: 0, DOGS: 1}
    
    # Create an empty set of data to fill.
    training_data = []
    
    # Balancing is very important in neural networks.
    catcount = 0
    dogcount = 0
    
    # A Function to create the trainging data (gather the data)
    def make_training_data(self):
        # Iterate over labels (cats/dogs)
        for label in self.LABELS:
            print(label)
            # Iterate over the two directories
            for f in tqdm(os.listdir(label)): # tqdm gives us a progress bar.
                try:
                    # Create a path to each file "f"
                    path = os.path.join(label, f)
                    # Convert each file to grayscale ("Is color a defining feature when contrasting cats & dogs?")
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # Convert each file (picture) to 50x50
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # Converting the data into a one-hot vector.
                    # Where: np.eye(X) Creates an X by X Identity matrix.
                    # And np.eye(X)[Y] Creates a 1 by X where the Yth element is "hot" (1).
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    # Get the count. If there are too many of one, we need to balance them.
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass # There are some image errors likely to appear.
        
        np.random.shuffle(self.training_data)
        np.save("trainging_data.npy", self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ", self.catcount)

if REBUILD_DATA:
    dogsvcats = DogsvsCats()
    dogsvcats.make_training_data()


# In[4]:


# Load our training_data from the npy file we created.
training_data = np.load("training_data.npy", allow_pickle = True)
# Show how many samples we have.
print(len(training_data))

import matplotlib.pyplot as plt
# Show our image in grayscale.
plt.imshow(training_data[1][0], cmap = "gray")
plt.show()


# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the network:
class Net(nn.Module):
    def __init__(self):
        # Must call super().__init__() for this to function.
        super().__init__()
        # Conv2d() --> 2 dimentional convolutional layer.
        # (1, 32, 5) --> [1 = input, 32 = output, 5 = KERNAL size]
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        # Since we cannot for sure know what to input for our first fully connected layer (fc1),
        # We need to test a random piece of data in.
        # -1 means we are taking any size shape, and we are viewing it as a 1x50x50 (a tensor object)
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        # The first pass (when setting up the NN) of convs(x) determines the shape we need to pass into fc1
        # and assigns it to self._to_linear.
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)
        
        
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
#         print(x[0].shape)
        if self._to_linear is None: # Meant to run one time (only for our 1 size batch)
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] # Grab the shape
        return x
    
    def forward(self, x):
        x = self.convs(x) # We can call this again, b/c the if statement won't run anymore.
        x = x.view(-1, self._to_linear) # We need to know the shape here too.
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim = 1)
    
net = Net()


# In[6]:


import torch.optim as optim

# Define our optimizer.
optimizer = optim.Adam(net.parameters(), lr = 0.001)
# Define our loss function.
loss_function = nn.MSELoss() # Mean Squared Error Loss (For our hot vector)

# We want to separate our X's and Y's.
# i[0] gives us X's only.
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
# Scale the imagery (from 0-255 to 0-1)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1 # Valuation size (10%)
val_size = int(len(X)*VAL_PCT)
print(val_size) # 2494

train_X = X[:-val_size] # From beginning to -val_size.
train_y = y[:-val_size]

test_X = X[-val_size:] # From -val_size onward.
test_y = y[-val_size:]

print(len(train_X)) # 22452 all batches
print(len(test_X)) # 2494 (~10 %)


# In[7]:


BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    # For i in tqdm(range from 0 to how many in train_X using BATCH_SIZE of 100)
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        # print(i, i + BATCH_SIZE) Prints each batch (0-100, 100-200, etc)
        batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i + BATCH_SIZE]
        
        # Zero the gradience.
        net.zero_grad()
        
        # Run the data through the network.
        outputs = net(batch_X)
        
        # Calculate the loss
        loss = loss_function(outputs, batch_y)
        
        # Back propagate (calculate weight changes)
        loss.backward()
        
        # Apply weight changes to layers.
        optimizer.step()

print(loss)


# In[8]:


correct = 0
total = 0
# To evaluate the testing data only (no learning from this)
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
        
print("Accuracy: ", round(correct/total, 3))

