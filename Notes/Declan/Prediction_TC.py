################################## Training and Validation Setup ##################################
import logging
import torch
import io
from random import seed
from random import randint
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from tqdm import tqdm


NGRAMS = int(input("Enter NGRAMS: "))

BATCH_SIZE = int(input("Enter Batch Size: "))

EMBED_DIMMENSION = int(input("Enter Embed Dim: "))

INITIAL_LEARNING_RATE = float(input("Enter Initial Learning Rate: "))


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
	# train_csv_path = root + 'train.csv'
	# test_csv_path = root + 'test.csv'
	while True:
		train_csv_path, test_csv_path = splitter(data, root)
		if vocab is None:
			logging.info('Building Vocab based on {}'.format(train_csv_path))
			vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
			# vocab.load_vectors(vectors='glove.6B.100d')
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
		if len(train_labels ^ test_labels) == 0:
			break
			# raise ValueError("Training and test labels don't match")
	return (TextClassificationDataset(vocab, train_data, train_labels),
		TextClassificationDataset(vocab, test_data, test_labels))

def splitter(data, root):
	train_csv_path = root + 'train.csv'
	test_csv_path = root + 'test.csv'
	train_file = open(train_csv_path, 'w')
	test_file = open(test_csv_path, 'w')

	line_count = len(open(data).readlines())
	seed(time.time())
	lines = set()

	while len(lines) < line_count * .1:
		lines.add(randint(0,line_count))

	with open(data) as data_file:
		for lineno, line in enumerate(data_file):
			if lineno in lines:
				test_file.write(line)
			else:
				train_file.write(line)
	train_file.close()
	test_file.close()
	return train_csv_path, test_csv_path

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
	        initrange = 1.5
	        self.embedding.weight.data.uniform_(-initrange, initrange)
                # Randomizes the initial weights uniform to (-initrange to initrange)
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

from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

Biomez_label = {1: "Biological",
                2: "Psychological",
                3: "Social"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

############################################# "Main" ##############################################

root = './.data/BIOME-z/'	

if not os.path.isdir(root):
	os.mkdir('./.data')
	os.mkdir(root)

data_file = root + 'data.csv'

# User input parameters: ngrams, batch size, embed dim, initial learning rate.

train_dataset, test_dataset = _setup_datasets(data=data_file, root=root, ngrams=NGRAMS, vocab=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # I am using the CPU in this case.

# Set variables for testing and learning.
VOCAB_SIZE = len(train_dataset.get_vocab())
vocab = train_dataset.get_vocab()

# Output size for the first layer.
NUM_CLASS = len(train_dataset.get_labels())

# Create the model (initialize weights and layers).
model = TextSentiment(VOCAB_SIZE, EMBED_DIMMENSION, NUM_CLASS).to(device)

EPOCHS = 25

min_valid_loss = float('inf')

# Criterion uses a combination of LogSoftmax() and NLLLoss()
criterion = torch.nn.CrossEntropyLoss().to(device)

# SGD uses stochastic gradient descent method.
optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE)

# Used to change learning rate (Descending Learning Rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ =     random_split(train_dataset, [train_len, len(train_dataset) - train_len])

example_text1 = "Genome editing technologies have been revolutionized by the discovery of prokaryotic RNA-guided defense system called CRISPR-Cas. Cas9, a single effector protein found in type II CRISPR systems, has been at the heart of this genome editing revolution. Nearly half of the Cas9s discovered so far belong to the type II-C subtype but have not been explored extensively. Type II-C CRISPR-Cas systems are the simplest of the type II systems, employing only three Cas proteins. Cas9s are central players in type II-C systems since they function in multiple steps of the CRISPR pathway, including adaptation and interference. Type II-C CRISPR systems are found in bacteria and archaea from very diverse environments, resulting in Cas9s with unique and potentially useful properties. Certain type II-C Cas9s possess unusually long PAMs, function in unique conditions (e.g., elevated temperature), and tend to be smaller in size. Here, we review the biology, mechanism, and applications of the type II-C CRISPR systems with particular emphasis on their Cas9s."
example_text2 = "The conventional description of Abraham Maslow's (1943, 1954) hierarchy of needs is inaccurate as a description of Maslow's later thought. Maslow (1969a) amended his model, placing self-transcendence as a motivational step beyond self-actualization. Objections to this reinterpretation are considered. Possible reasons for the persistence of the conventional account are described. Recognizing self-transcendence as part of Maslow's hierarchy has important consequences for theory and research: (a) a more comprehensive understanding of worldviews regarding the meaning of life; (b) broader understanding of the motivational roots of altruism, social progress, and wisdom; (c) a deeper understanding of religious violence; (d) integration of the psychology of religion and spirituality into the mainstream of psychology; and (e) a more multiculturally integrated approach to psychological theory."
example_text3 = "Purpose : A subgroup of principals—leaders for social justice—guide their schools to transform the culture, curriculum, pedagogical practices, atmosphere, and schoolwide priorities to benefit marginalized students. The purpose of the article is to develop a theory of this social justice educational leadership.Research Design: This empirical study examined the following questions: (a) In what ways are principals enacting social justice in public schools? (b) What resistance do social justice—driven principals face in their justice work? (c) What strategies do principals develop to sustain their ability to enact social justice in light of the resistance they face in public schools?Data Collection and Analysis: A critical, qualitative, positioned-subject approach combined with principles of autoethnography guided the research methods. Seven public school leaders who came to the principalship with a social justice orientation, who make issues of race, class, gender, disability, sexual orientation, and other historically marginalizing factors central to their advocacy, leadership practice, and vision, and who have demonstrated success in making their schools more just, were studied through interviews.Findings:A description of (a) how the principals enacted social justice, (b) the resistance they faced as well as the toll the resistance had on them, and (c) the strategies they developed to sustain their social justice work is provided in detail. Implications for administrator preparation are made at the close of this article."
example_text4 = "Social psychologists have established various psychological mechanisms that influence perception of risk and compliance in general. The empirical investigation in this paper focused on how those mechanisms apply to complying with scams. A scale of susceptibility to persuasion was developed, validated and then applied to the phenomena of scam compliance in two studies. In the first study participants answered questions on the susceptibility to persuasion scale and a series of questions about lifetime compliance with 14 fraudulent scenarios. The scale was factorised and tested for reliability. Four reliable factors contributed to susceptibility to persuasion: influence of authority, social influence, self-control and the need for consistency. The susceptibility to persuasion scale was then used to predict overall lifetime scam compliance. Social influence, the need for consistency and self-control all had an impact on universal scam compliance. In the second study an independent sample of participants filled out the susceptibility to persuasion scale and answered questions measuring scam compliance for the past three years across nine fraudulent scenarios. The susceptibility to persuasion scale was validated and confirmed. Scam compliance over the past three years was measured and the results showed that authority, social influence, the need for consistency and self-control all informed scam compliance over that period."

# Create an automation for optimizing the parameters (Ngrams, Embed_dim, batch size, etc)
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

print("This is a %s related article." %Biomez_label[predict(example_text1, model, vocab, NGRAMS)])
print("This is a %s related article." %Biomez_label[predict(example_text2, model, vocab, NGRAMS)])
print("This is a %s related article." %Biomez_label[predict(example_text3, model, vocab, NGRAMS)])
print("This is a %s related article." %Biomez_label[predict(example_text4, model, vocab, NGRAMS)])
