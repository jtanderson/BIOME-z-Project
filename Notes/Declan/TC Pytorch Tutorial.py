#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Text-classification tutorial.
# Uses torchtext's datasets.
import os
import torch
import torchtext
from torchtext.datasets import text_classification

NGRAMS = 2 # How it groups the words together
# NGRAMS = 3 --> 'Apples are healthy'
# NGRAMS = 2 --> 'Apples are'

# If ./data does not exist, create a dir for it.
if not os.path.isdir('./.data'):
    os.mkdir('./.data')

# Download the datasets for both training and testing respectively.
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root = './.data', ngrams=NGRAMS, vocab=None)

# The size of the batch.
BATCH_SIZE = 16


# In[15]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # I am using the CPU in this case.
# Creates a copy of the Tensor and formats it for either the GPU (cuda) or CPU.


# In[16]:


import torch.nn as nn
import torch.nn.functional as F

class TextSentiment(nn.Module):
    # Initialization
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        # Set "embedding" to a new instance of "EmbeddingBag"
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # Set fc to a linear transformation of embed_dim(input), and num_class(output)
        self.fc = nn.Linear(embed_dim, num_class)
        # Call the function to initialize weights.
        self.init_weights()
    
    # Initializes the weights of the layer.
    # Another method is "Encoding & Decoding":
    # Instead of associating different words with one another in a dense structure (Embedding),
    # it takes assigns a placement of a unique word in a vector.
    
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
  
    # Slitches all layers together.
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# In[19]:


# There are FOUR categories that the AG_NEWS has:
# World, Sports, Business, Sci/Tec
# Our BIOME-Z project will have 12 individual labels Ex: intermediate-social.

# Set variables for testing and learning.
VOCAB_SIZE = len(train_dataset.get_vocab()) # 1,308,844

# Input size for first layer.
EMBED_DIM = 32

# Output size for the first layer.
NUM_CLASS = len(train_dataset.get_labels())

# Create the model (initialize weights and layers).
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

# A custom function to create batch sizes given varying text sizes.
def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    
    # Returns cumulative sum given the dimension.
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


# In[23]:


from torch.utils.data import DataLoader

# Function for training the model.
def train_func(sub_train_):
    train_loss = 0
    train_acc = 0
    
    # Load the data in: (dataset, batchsize, shuffle, collate_fn)
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                     collate_fn=generate_batch)
    
    
    for i, (text, offsets, cls) in enumerate(data):
        
        # Zeros our gradience (related to loss)
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        
        # Run out data through our model.
        output = model(text, offsets)
        
        # Get the loss value.
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward() # Back propagate (calculate weight change).
        optimizer.step() # Apply weight changes.
        train_acc += (output.argmax(1) == cls).sum().item()
        
    # Adjust learning rate:
    scheduler.step()
    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()
            
    return loss / len(data_), acc / len(data_)
    


# In[25]:


import time
from torch.utils.data.dataset import random_split

EPOCHS = 5

min_valid_loss = float('inf')

# Criterion uses a combination of LogSoftmax() and NLLLoss()
criterion = torch.nn.CrossEntropyLoss().to(device)

# SGD uses stochastic gradient descent method.
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)

# Used to change learning rate (Descending Learning Rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
# StepLR(optimizer -> The optimizer it wraps
#        step_size -> The period of learning rate decay
#        gamma     -> Multiplicative factor of learning rate decay
#        last_epoch-> The index of the last epoch).

# lr = 4.0    if epoch < 1
# lr = 3.6    if 1 <= epoch < 2
# lr = 3.24   if 2 <= epoch < 3
# lr = 2.916  if 3 <= epoch < 4
# lr = 2.6244 if 4 <= epoch < 5 Understand?


train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ =     random_split(train_dataset, [train_len, len(train_dataset) - train_len])

# For each epoch:
for epoch in range(EPOCHS):
    
    # Calculate start time.
    start_time = time.time()
    
    # Run training and testing.
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)
    
    # Calculate time values.
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    
    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


# In[ ]:


# Ending notes:
# What we need to start using Pytorch and the BIOME-z data:
# > Our set of data (Abstract & Labels) in a testing set (?) and obviously training set.
# > Learn how to use torchtext to load our data in.
# > How to properly use EmbeddingBag (if we are using this).
# Decisions on:
# - Ngrams (how we cluster the words).
# - Epochs (how many times we want to run over the data for training).
# - If we are going to have a static or dynamic learning rate.
# - Which optimizer we will be using (there are quite a few).
# I am sure a few other things...

