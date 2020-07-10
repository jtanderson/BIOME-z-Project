# Jack Stoetzel

###################################### RDF to CSV Converter #######################################
import time
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, FOAF
from rdflib import Literal, URIRef, XSD

class paper:
	def __init__(self):
		self.subject = URIRef("")
		self.title = ""
		self.abstract = ""
		self.bio = 0
		self.psych = 0
		self.social = 0

def isConnected(graph):
	print("Enter at", time.ctime())
	start = time.time()
	if graph.connected():
		end = time.time()
		print("Exit at", time.ctime(), '(', end - start, ") - Not Connected")
		# exit()
	else:
		end = time.time()
		print("Exit at", time.ctime(), '(', end - start, ") - Connected")

def replacer(words):
	# Need to replace \n with \ and " with '
	words = words.replace('\n', " \\ ")
	words = words.replace('\"', '\'')
	return words

def insert(collection, subject, object, category):
	found = False
	marker = -1
	for i in range(len(collection)):
		if collection[i].subject == subject:
			found = True
			marker = i
	if not found:
		obj = paper()
		obj.subject = subject
		marker = len(collection)
		collection.append(obj)
	# print (marker)
	if category == 'a':
		collection[marker].abstract = replacer(str(object))
	elif category == 't':
		collection[marker].title = replacer(str(object))
	elif category == 'b':
		collection[marker].bio = 1
	elif category == 'p':
		collection[marker].psych = 2
	elif category == 's':
		collection[marker].social = 3
	return collection

# Converts the .rdf file to a .csv file
def parser(data_file='./data.csv'):
	#file = 'tester.rdf'
	file = '../../BIOME-z.rdf'
	output = open(data_file, 'w')

	graph = rdflib.Graph()

	start = time.time()

	#Parses the .rdf files and stores it into the graph
	graph.parse(file)

	end = time.time()

	print("{0:.2f} seconds to parse BIOME-z.rdf".format((end - start)))

	# A test to see if the graph made by the .parse() function is connected (It is NOT)
	# isConnected(graph)

	# Some strings to look for when traversing through the graph
	TITLE = "http://purl.org/dc/elements/1.1/title"
	ABSTRACT = "http://purl.org/dc/terms/abstract"
	SUBJECT = "http://purl.org/dc/elements/1.1/subject"

	# For some reason, I think these only work as literals
	BIO = Literal("biological")
	PSYCH = Literal("psychological")
	SOCIAL = Literal("social")

	# Making a set of subject URIRef ID's. These will be used later, but we dont want repeats
	collection = []

	counter = -1
	domain_count = [0, 0, 0]

	# This will find the URIRef ID for every paper that has an abstract attached to it
	start = time.time()
	for s, p, o in graph:
		if ABSTRACT in p:
			collection = insert(collection, s, o, 'a')
			counter += 1
		if TITLE in p:
			collection = insert(collection, s, o, 't')
		if SUBJECT in p and BIO in o:
			collection = insert(collection, s, o, 'b')
			domain_count[0] += 1
		if SUBJECT in p and PSYCH in o:
			collection = insert(collection, s, o, 'p')
			domain_count[1] += 1
		if SUBJECT in p and SOCIAL in o:
			collection = insert(collection, s, o, 's')
			domain_count[2] += 1

	end = time.time()

	print("{0:.2f} seconds to ID {1:d} articles".format((end - start), counter))
	print("Bio    -", domain_count[0])
	print("Psych  -", domain_count[1])
	print("Social -", domain_count[2])

	for obj in collection:
			if obj.social != 0:
				output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.social), str(obj.title), str(obj.abstract)))
			elif obj.psych != 0:
				output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.psych), str(obj.title), str(obj.abstract)))
			elif obj.bio != 0:
				output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.bio), str(obj.title), str(obj.abstract)))
			
			

	output.close()

################################## Training and Validation Setup ##################################
import logging
import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from tqdm import tqdm

def _csv_iterator(data_path, ngrams, yield_cls=False):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = ' '.join(row[1:])
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)

def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                        for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)

class TextClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, vocab, data, labels):
        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab


    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab

def _setup_datasets(data, root, ngrams=1, vocab=None, include_unk=False):
	
	# Need to split data file into train.csv and test.csv
	train_csv_path = root + 'train.csv'
	test_csv_path = root + 'test.csv'

	if vocab is None:
		logging.info('Building Vocab based on {}'.format(train_csv_path))
		vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))

	else:
		if not isinstance(vocab, Vocab):
			raise TypeError("Passed vocabulary is not of type Vocab")

	logging.info('Vocab has {} entries'.format(len(vocab)))

	logging.info('Creating training data')
	train_data, train_labels = _create_data_from_iterator(
		vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)

	logging.info('Creating testing data')
	test_data, test_labels = _create_data_from_iterator(
		vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)

	# print(train_labels, "^", test_labels, "=", len(train_labels ^ test_labels))
	# exit()
	if len(train_labels ^ test_labels) > 0:
		raise ValueError("Training and test labels don't match")

	return (TextClassificationDataset(vocab, train_data, train_labels),
		TextClassificationDataset(vocab, test_data, test_labels))

############################### Text Classification with TorchText ################################
import os
import torch
import torchtext
from torchtext.datasets import text_classification

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import time
from torch.utils.data.dataset import random_split

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

# Declans classifier program 
def classifier(data_file='./data.csv', root='./'):

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

	NGRAMS = 2 # How it groups the words together
	
	train_dataset, test_dataset = _setup_datasets(data=data_file, root=root, ngrams=NGRAMS, vocab=None)

	BATCH_SIZE = 16

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # I am using the CPU in this case.

	# Set variables for testing and learning.
	VOCAB_SIZE = len(train_dataset.get_vocab()) # 1,308,844

	# Input size for first layer.
	EMBED_DIM = 32

	# Output size for the first layer.
	NUM_CLASS = len(train_dataset.get_labels())

	# Create the model (initialize weights and layers).
	model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

	EPOCHS = 24

	min_valid_loss = float('inf')

	# Criterion uses a combination of LogSoftmax() and NLLLoss()
	criterion = torch.nn.CrossEntropyLoss().to(device)

	# SGD uses stochastic gradient descent method.
	optimizer = torch.optim.SGD(model.parameters(), lr=4.0)

	# Used to change learning rate (Descending Learning Rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.8)

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

	print('Checking the results of test dataset...')
	test_loss, test_acc = test(test_dataset)
	print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

############################################# "Main" ##############################################

root = './.data/BIOME-z/'	

if not os.path.isdir(root):
    os.mkdir(root)

data_file = root + 'data.csv'

parser(data_file)
classifier(data_file, root)
