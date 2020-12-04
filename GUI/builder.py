import converter
from datasets import _setup_datasets
import classification as train

import time
import torch

import re
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator

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

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass



def builder(folder, NGRAMS, GAMMA, BATCH_SIZE, LEARNING_RATE, EMBED_DIM, EPOCHS, Progress, Context):

    Statistics = stats_data()

    root = './.data/' + folder + '/'
    data_file = root + 'data.csv'
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

    print("NGRAMS \t BATCH_SIZE \t EMBED_DIM \t LEARNING_RATE \t STEP_SIZE \t GAMMA \n")
    print(f"{NGRAMS} \t {BATCH_SIZE} \t\t {EMBED_DIM} \t\t {LEARNING_RATE} \t\t {STEP_SIZE} \t\t {GAMMA} \n")
     	
    train_dataset, test_dataset = _setup_datasets(data=data_file, root=root, ngrams=NGRAMS, vocab=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on {device}. \n")

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

    cat = [0, 0, 0]
    for i in range(train_len):
        if train_dataset[i][0] == 0:
            cat[0]+=1
        elif train_dataset[i][0] == 1:
            cat[1]+=1
        else:
            cat[2]+=1

    print(cat)

    # This will make valid training sets until the distribution of domains is equal

    sub_train_, sub_valid_ = train.traingingSplit(train_dataset, train_len, BATCH_SIZE, len(categories))
    
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
        
        
        # Calculate time values.
        secs = int(time.time() - epoch_start_time)
        mins = secs / 60
        secs = secs % 60

        print(f'Epoch: {epoch + 1} | time in {mins:.0f} minutes, {secs:.1f} seconds')
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        Progress.set(Progress.get() + 1.0/EPOCHS * 100)
        Context.update()

    # Calculate time values.
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Checking the results of test dataset after %d minutes, %d seconds and %d Epochs'%(mins, secs, EPOCHS))

    test_loss, test_acc = train.test(test_dataset, BATCH_SIZE, device, model, criterion, categories)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test) \n\n\n')
    # Saves the model
    torch.save(model, root + "model")
    return Statistics