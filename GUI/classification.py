############################### Text Classification with TorchText ################################
import datasets

import os
import time
import torch
import pprint
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import text_classification


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
	    
	    def init_weights(self):
	        initrange = 0.5
	        self.embedding.weight.data.uniform_(-initrange, initrange)
	        self.fc.weight.data.uniform_(-initrange, initrange)
	        self.fc.bias.data.zero_()
	  
	    # Slitches all layers together.
	    def forward(self, text, offsets):
	        embedded = self.embedding(text, offsets)
	        return self.fc(embedded)

# A custom function to create batch sizes given varying text sizes.
def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    
    # Returns cumulative sum given the dimension.
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

# Function for training the model.
def train_func(sub_train_, BATCH_SIZE, optimizer, device, model, criterion, scheduler):
    train_loss = 0
    train_acc = 0
    
    # Load the data in: (dataset, batchsize, shuffle, collate_fn)
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    
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

def test_valid(data_, BATCH_SIZE, device, model, criterion):
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

def test(data_, BATCH_SIZE, device, model, criterion, categories):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    
    answers = []
    correct = []
    for i in range(len(categories)):
    	answers.append(0.0)
    	correct.append(0.0)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()    
            acc += (output.argmax(1) == cls).sum().item()

            accuracy = (output.argmax(1) == cls)
          
            # print(cls)
            for i in range(len(cls)):
            	answers[(cls[i])] += 1
            	if accuracy[i] == True:
            		correct[(output.argmax(1)[i])] += 1
            		
    for i in range(len(categories)):
    	print(f'{categories[i]}: \t {(correct[i]/answers[i]) * 100:.1f}%  \t ({correct[i]}/{answers[i]})')
    
    print()
    return loss / len(data_), acc / len(data_)

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def traingingSplit(train_dataset, train_len, BATCH_SIZE, catCount):
	balanced = False
	while not balanced:
		valid_domain = []
		for i in range(catCount):
			valid_domain.append(0)

		sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
		data = DataLoader(sub_valid_, batch_size=BATCH_SIZE, collate_fn=generate_batch)	

		for text, offsets, cls in data:
			equal = len(cls) / catCount
			for i in cls:
				valid_domain[i] += 1

		balanced = True
		for i in valid_domain:
			if i != round(equal) and i != (round(equal)+1):
				balanced = False

	return sub_train_, sub_valid_

