#!/usr/bin/env python
# coding: utf-8

# In[49]:


import torch
import torchvision
import torch.nn as nn # Object Oriented Programming Tools.
import torch.nn.functional as F # Functions.
from torchvision import transforms, datasets

# Grabbing the datasets to learn off of
train = datasets.MNIST("", train=True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                     transform = transforms.Compose([transforms.ToTensor()]))

# Loading the data given which type, the size of the batch sent, and whether it is shuffled.
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

# |--------------------------------------------------------------------|
# for data in trainset:
#     print (data)
#     break

# x, y = data[0][0], data[1][0]
# print(y)

# import matplotlib.pyplot as plt

# print(data[0][0].shape)
# # Shows us the size of our data:
# # A 28x28 pixel image

# # Next showing the data using [view as a 28x28]
# plt.imshow(data[0][0].view(28, 28))
# # This shows us the actual image of a "0-9"
# plt.show()


# total = 0
# counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

# for data in trainset:
#     Xs, Ys = data
#     for y in Ys:
#         counter_dict[int(y)] += 1
#         total+=1
        
# print(counter_dict)

# for i in counter_dict:
#     print(f"{i}: {counter_dict[i]/total*100}")
#     # Shows the distribution of different numbers appearing in the MNIST dataset.
# |--------------------------------------------------------------------|

# Building our neural network:
class Net(nn.Module):
    def __init__(self): # Initialization function:
        super().__init__() # Super corresponds to nn.Module, so we are initializing itself. A Must Need
        # Setting up our first fully connected layer (fclayer):
        # Our target is 3 layers of [[64]] NEURONS (See 64 below).
        # nn.Linear(input, output)
        self.fc1 = nn.Linear(28*28, 64) # Where 28*28 is the INPUTTED data itself (For Mnist, we have a flat 28*28) & 64 Neuron Output.
        # Linear = Means fully connected (flat and simple)
        self.fc2 = nn.Linear(64, 64) # TAKES IN 64 (FROM FC1)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) # OUTPUTS 10 (RESULTS ARE 0 - 9).
        
    # Forward function (how we want data to pass through).
    def forward(self, x):
        # F.relu is out activation function, we need this to learn more effectively.
        # relu = rectifiedLinear (A ramp function to start the learning) (Whether the neuron is firing or not).
        # Keeps the return from loss "explosion"
        x = F.relu(self.fc1(x)) # Firstly, "x" passes through all fully connected layers (1-4).
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        # We want a probability distribution as our output:
        # Example [Cats, Dogs, Human] -> [0.10, 0.05, 0.85] If we pass a human in as data.
        # We are using Log-Softmax (Since we are using a multi-class (0-9 results))
        return F.log_softmax(x, dim=1) # dim=1 which dim it will work on.
        # Example: A tensor with dimensions: 3 x 5,
        # log_softmax(x, dim=0) makes the sum of the tensors with 3 numbers. (1, 1, 1)
        # log_softmax(x, dim=1) makes the sum of the tensors with 5 numbers. (1, 1, 1, 1, 1)
        
        
net = Net()
print(net)

# "Simple Neural Network": Feed forward network (Goes from one side to another).


# |--------------------------------------------------------------------|
# Now we are passing data through the network:

# X = torch.rand((28, 28)) # Passing a random 28 x 28 tensor. (NOT THE ACTUAL DATA)
# X = X.view(-1, 28*28) # "-1" Specifies that the input is an unknown dimension.
# output = net(X) # Running it through the layers, and outputting a prediction.

# output # Prints out predictions.

# Training the model...
# Two concepts: "Loss" & "Optimizer"
# Loss: The measure of how wrong the model is. GOAL: Have the loss decrease.
# Optimizer: Needs to go through and adjust the "weights" in order to reduce loss, depending on learning rate.
# |--------------------------------------------------------------------|


import torch.optim as optim

# net.parameters(): What (weight) is adjustable by this optimizer.
# lr=0.01: The learning rate. Should not be TOO HIGH or TOO LOW. Think of this as the size of a step.
# Decaying learning rate: Starts with larger rate, then goes to a smaller learning rate (for complex networks)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# We want to iterate over all of our data and through our model.
# EPOCHS = number of full passes through our entire dataset.
EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # Data is in a batch of featuresets and labels.
        X, Y = data
        # print(X[0]) Would print out the image (rows of pixels)
        # Everytime before we pass data through our neural network, we want to run net.zero_grad()
        # -> We do not want to pass the entire batch size usually.
        net.zero_grad() # Starts at zero with gradiance (which contains our loss).
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, Y) # Calculates loss.
        # Two ways to calculate loss:
        #     If your input is a scalar vector (Like in our data), use: nll_loss.
        #     If your data is a one-hot vector EX: [0, 0, 1], use mean_squared error.
        # Now we back-propagate: Which fine-tunes our weights properly based on our loss, in order to increase generalization.
        loss.backward() # Back propagation.
        optimizer.step() # Actually changes the weights.
    print(loss) # Prints out loss. It SHOULD lower each time it occurs.
        


# In[54]:


# Calculating how correct it is:

correct = 0
total = 0

with torch.no_grad(): # We do not want to optimize (just check how well it is)
    for data in trainset:
        X, Y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == Y[idx]:
                correct += 1
            total += 1
            
print("Accuracy: ", round(correct/total, 3))


# In[56]:


import matplotlib.pyplot as plt
plt.imshow(X[0].view(28, 28))
plt.show() # Prints out the image.
print(torch.argmax(net(X[0].view(-1, 784))[0])) # Prints out the prediction value.
# Hopefully they should be the same.
# There is a library called "ignite" which helps in training loops.

