#!/usr/bin/env python
# coding: utf-8

# In[9]:


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

title_field = data.Field(sequential=True,
                        lower=True,
                        include_lengths=True,
                        use_vocab=True)

training_fields = [
    ('Item', None), # Not needed for any classification.
    ('Classify', classify_field), # Binary classification.
    ('Title', title_field), # Processed as text.
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
desc_field.build_vocab(trainData, validData, max_size=5000, min_freq=1, vectors='glove.6B.100d')
title_field.build_vocab(trainData)

# The data has definitely loaded into tensors, however; unsure what is next.
# Embedding bag?
# Create bucket iterator?
# Create custom batch generator.


# In[ ]:




