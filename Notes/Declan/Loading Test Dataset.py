#!/usr/bin/env python
# coding: utf-8

# In[28]:


import re
import tqdm
import spacy
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
# [Number] [Classification] [Title] [Abstract/Description]
desc_field = data.Field(sequential=True,
                        lower=True,
                        tokenize=tokenizer,
                        include_lengths=True,
                        use_vocab=True)

classify_field = data.Field(sequential=False,
                           lower=False,
                           include_lengths=False,
                           use_vocab=False)


training_fields = [
    ('Item', None), # Not needed for any classification.
    ('Classify', classify_field), # Binary classification.
    ('Title', desc_field), # Processed as text.
    ('Abstract', desc_field) # Also processed as text.
]

# Assigning datasets.
trainData, validData = data.TabularDataset.splits(path='./data',
                                       format='csv',
                                       train='train.csv',
                                       validation='valid.csv',
                                       fields=training_fields,
                                       skip_header=False)

# Now that we have the dataset(s), we can grab data using:
# ex = trainData[0]
# ex.Item  --> '1'
# ex.Title --> ['Beyond', 'Mead', 'Symbolic', 'Interaction', ... ] (Article title tokenized).

# Creating the vocabulary:
desc_field.build_vocab(trainData, validData, max_size=5000, min_freq=1)
# vectors='glove.6B.100d'

# The data has definitely loaded into tensors, however; unsure what is next.
# Embedding bag?
# Create bucket iterator?
# Create custom batch generator.

trainIterator, testIterator = data.BucketIterator.splits(
    (trainData, validData),
    batch_size=2,
    device="cpu",
    sort_key=lambda x: len(x.Abstract),
    sort_within_batch=False,
    repeat=False)


# Testing whether it loaded in properly

trainData[0] # Torchtext.data.example.Example

for batch in trainIterator: # Prints out the tensors for the Classify and Title.
    print(batch.Classify)
    print(batch.Title)
    
trainData[0].__dict__.keys() # Prints out the "keys" as in "Primary Key"
trainData[0].Abstract[:5] # Prints out 5 words from the first abstract.

# Creating a Batch Wrapper in order to convert the batch into a tuple.
class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars
        
    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var) # Only one input (for us, probably not.)
            if self.y_vars is None:
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))
            yield(x, y)
            
    def __len__(self):
        return len(self.dl)
    
train_dl = BatchWrapper(trainIterator, "Abstract", ["Classify"])
test_dl = BatchWrapper(testIterator, "Abstract", ["Classify"])
print(train_dl)


# In[ ]:




