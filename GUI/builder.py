import converter
from datasets import _setup_datasets
import classification as train

import re
import math
import time
import torch

from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator


# This small class is meant to hold run data.
# It will then be passed into the Tk gui to be graphed.
class stats_data:
	def __init__(self):
		super().__init__()
		self.train_loss = []
		self.valid_loss = []
		self.train_acc = []
		self.valid_acc = []
		self.epochs = []

		self.train_size = 0
		self.test_size = 0
		self.labels = []
		self.sizes = []

		self.vocab_size = 0
		self.test_loss = 0
		self.test_acc = 0
		self.time_min = 0
		self.time_sec = 0

		self.initlrn = 0.00
		self.gamma = 0.00
		self.epoch = 0
		self.ngram = 0
		self.batch = 0
		self.embed = 0

		self.train_cat_label = []
		self.train_cat_val = []
		self.test_cat_label = []
		self.test_cat_val = []

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		pass


def builder(folder, NGRAMS, GAMMA, BATCH_SIZE, LEARNING_RATE, EMBED_DIM, EPOCHS, Progress, Context):
	Statistics = stats_data()
	Statistics.initlrn = LEARNING_RATE
	Statistics.gamma = GAMMA
	Statistics.epoch = EPOCHS
	Statistics.ngram = NGRAMS
	Statistics.batch = BATCH_SIZE
	Statistics.embed = EMBED_DIM

	root = './.data/' + folder + '/'
	#data_file = 'data.csv'
	#labels = open('labels.txt', 'r')
	data_file = root + 'data.csv' #Might need to be this not sure at the moment
	print(root + 'labels.txt')
	labels = open(root+'labels.txt', 'r')

	categories = []

	for line in labels:
		categories.append(line.replace('\n', ''))

	labels.close()

	categories.sort(reverse=False)

	# Ngrams, Batchsize, Embeddim, init lrn rate, gamma, epochs are acquired from the gui.
	INIT_WEIGHT = .1
	STEP_SIZE = 1

	# Calculate start time.
	start_time = time.time()
		
	train_dataset, test_dataset = _setup_datasets(data=data_file, root=root, ngrams=NGRAMS, vocab=None)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Set variables for testing and learning.
	VOCAB_SIZE = len(train_dataset.get_vocab())

	# Output size for the first layer.
	NUM_CLASS = len(train_dataset.get_labels())

	# Create the model (initialize weights and layers).
	model = train.TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

	min_valid_loss = float('inf')

	# Criterion uses a combination of LogSoftmax() and NLLLoss()
	criterion = torch.nn.CrossEntropyLoss().to(device)

	# SGD uses stochastic gradient descent method.
	optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

	# Used to change learning rate (Descending Learning Rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEP_SIZE, gamma=GAMMA)

	train_len = int(len(train_dataset) * 0.95)

	for label in categories: Statistics.labels.append(label)
	Statistics.train_size = train_len
	Statistics.test_size = len(train_dataset) - train_len

	cat = [0 for i in range(len(categories))]
	for x in range(train_len):
		for y in range(len(categories)):
			if train_dataset[x][0] == y:
				cat[y] += 1

	# This will make valid training sets until the distribution of domains is equal
	sub_train_, sub_valid_ = train.trainingSplit(train_dataset, train_len, BATCH_SIZE, len(categories))
	
	num_epoch = 0
	# For each epoch:
	for epoch in range(EPOCHS):
		# sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
		# Calculate start time.
		epoch_start_time = time.time()
		
		# Run training and testing.
		train_loss, train_acc = train.train_func(sub_train_, BATCH_SIZE, optimizer, device, model, criterion, scheduler)
		valid_loss, valid_acc = train.test_valid(sub_valid_, BATCH_SIZE, device, model, criterion)

		Statistics.epochs.append(epoch + 1)
		Statistics.train_acc.append(train_acc)
		Statistics.valid_acc.append(valid_acc)
		Statistics.train_loss.append(train_loss)
		Statistics.valid_loss.append(valid_loss.item())

		Progress.set(Progress.get() + 1.0/EPOCHS * 100)
		Context.update()

	# Calculate time values.
	secs = int(time.time() - start_time)
	mins = secs / 60
	secs = secs % 60

	test_loss, test_acc, test_comp = train.test(test_dataset, BATCH_SIZE, device, model, criterion, categories)

	Statistics.vocab_size = VOCAB_SIZE
	Statistics.test_loss = round(test_loss.item(), 4)
	Statistics.test_acc = round(test_acc * 100, 3)
	Statistics.time_min = int(math.floor(mins))
	Statistics.time_sec = secs
	Statistics.train_cat_label = categories
	Statistics.train_cat_val = cat
	Statistics.test_cat_label = categories
	Statistics.test_cat_val = test_comp

	# Saves the model
	torch.save(model, root + "model")
	torch.save(train_dataset.get_vocab(), root+"vocab")
	return Statistics
