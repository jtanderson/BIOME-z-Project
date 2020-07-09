#!/usr/bin/env python
# coding: utf-8

# In[5]:


import re
import tqdm
import spacy
import torch
import torchtext
from torchtext import data

nlp = spacy.load('en')

# Define our tokenizer we will use in preprocessing.
# Tokenizing: separating the different words from each other.
# Ex: "A large potato database" --> "A" "large" "potato" "database"

def tokenizer(s):
    return [w.text.lower() for w in nlp(parsing(s))]

def parsing(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()

# Test csv file columns:
# [Classification] [Title] [Abstract]
desc_field = data.Field(sequential=True,
                        lower=True,
                        batch_first=True,
                        tokenize=tokenizer,
                        include_lengths=True,
                        use_vocab=True)

classify_field = data.LabelField(dtype=torch.float,
                           batch_first=True,
                           include_lengths=False,
                           use_vocab=False)

training_fields = [
    ('Label', classify_field),
    ('Title', desc_field),
    ('Abstract', desc_field)
]

trainData, validData = data.TabularDataset.splits(path='./data',
                                      format='csv',
                                      train='data.csv',
                                      validation='new_valid.csv',
                                      fields=training_fields,
                                      skip_header=False)



# Now that we have the dataset(s), we can grab data using:
# ex = trainData[0]
# ex.Item  --> '1'
# ex.Title --> ['Beyond', 'Mead', 'Symbolic', 'Interaction', ... ] (Article title tokenized).

# Creating the vocabulary:
desc_field.build_vocab(trainData, validData, max_size=5000, vectors='glove.6B.100d', min_freq=1)
classify_field.build_vocab(trainData)

# Loading torchtext's Iterator.
trainIterator, validIterator = data.BucketIterator.splits(
    (trainData, validData),
    batch_size=64, # A low number due to the smaller input data.
    device="cpu",
    sort_key=lambda x: len(x.Abstract),
    sort_within_batch=True,
    repeat=False)


# Testing whether it loaded in properly
# ----------------------------------------------------------------------
# trainData[0] # Torchtext.data.example.Example
# for batch in trainIterator: # Prints out the tensors for the Classify and Title.
#     print(batch.Label)
#     print(batch.Title)
#     print(batch.Abstract)
    
trainData[0].__dict__.keys() # Prints out the "keys" as in "Primary Key"
trainData[0].Abstract[:5] # Prints out 5 words from the first abstract.
ex = trainData[4]
print(ex.Title)
# ----------------------------------------------------------------------

# Creating a Batch Wrapper in order to convert the batch into a tuple.
# class BatchWrapper:
#     def __init__(self, dl, x_var, y_vars):
#         self.dl, self.x_var, self.y_vars = dl, x_var, y_vars
        
#     def __iter__(self):
#         for batch in self.dl:
#             x = getattr(batch, self.x_var) # Only one input (for us, probably not.)
#             if self.y_vars is None:
#                 y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
#             else:
#                 y = torch.zeros((1))
#             yield(x, y)
            
#     def __len__(self):
#         return len(self.dl)
    
# train_dl = BatchWrapper(trainIterator, "Abstract", ["Classify"])
# test_dl = BatchWrapper(testIterator, "Abstract", ["Classify"])


# In[6]:


# Now, perhaps we can "train" the model? There is NOT really enough data to learn, but 
# this will be a good setup for when we test Jack's 3-classification data sample.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Notes for the class __init__ arguments:
# vocab_size -> The size of the entire vocab, retreived by:
#      print(len(desc_field.vocab))
# embed_dim  -> The size of our first layer (likely: 10 or 32. Depends.)
# num_classs -> The number of labels we will be outputting:
#      My Cat/Not Cat classification: 2 output options.
#      Jack's classification demo: 3 output options 


class TextClassificationNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, hidden_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        # Create an embedding bag (which computes the mean value of a bag):
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        
        # Create the an LSTM layer.
        self.lstm = nn.LSTM(embed_dim,
                          num_class,
                          hidden_dim,
                          n_layers,
                          bidirectional=bidirectional,
                          dropout=dropout,
                          batch_first=True
                         )
        # Create our full connected layer.
        self.fc = nn.Linear(hidden_dim * 2, num_class)
        
        # Activation function.
        self.act = nn.Sigmoid()
    
#     def init_weights(self):
#         # Not sure what this range demarcates.
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc.weight.data.uniform_(-initrange, initrange)
#         self.fc.bias.data.zero()
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        # Packs a tensor containing sequences of variable length.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        # Returns a PackedSequence.
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        dense_output = self.fc(hidden)
        
        outputs = self.act(dense_output)
        
        return outputs
    

# Initialization:
vocab_size = len(desc_field.vocab)
embed_dim = 100
num_class = 1
hidden_dim = 32
# num_layers = 2
bidirectional = True
dropout = 0.2

model = TextClassificationNN(vocab_size, embed_dim, num_class, hidden_dim, 2, bidirectional, dropout)

# Initialize the pretrained embedding:
pretrained_embeddings = desc_field.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
print(pretrained_embeddings.shape)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

# Round to the nearest integer
def accuracy(pred, y):
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

model = model.to("cpu")
criterion = criterion.to("cpu")

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.Abstract
        predictions = model(text, text_lengths).squeeze()
        loss = criterion(predictions, batch.Label)
        acc = accuracy(predictions, batch.Label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            optimizer.zero_grad()
            text, text_lengths = batch.Abstract
            predictions = model(text, text_lengths).squeeze()
            loss = criterion(predictions, batch.Label)
            acc = accuracy(predictions, batch.Label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            return epoch_loss / len(iterator), epoch_acc / len(iterator)
    

EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, trainIterator, optimizer, criterion)
    
    valid_loss, valid_acc = evaluate(model, validIterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    printf(f'\tTrain Loss: {train_loss:3f} | Train Acc: {train_acc*100:.2f}%')
    printf(f'\tValid Loss: {valid_loss:3f} | Valid Acc: {valid_acc*100:.2f}%')

