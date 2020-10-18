import converter
from datasets import _setup_datasets
import classification as train

import time
import torch

import re
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator

############################################# "Main" ##############################################

def builder(folder):

    root = './.data/' + folder + '/'

    data_file = root + 'data.csv'

    labels = open(root+'labels.txt', 'r')

    categories = []

    for line in labels:
        categories.append(line.replace('\n', ''))

    categories.sort(reverse=False)

    # domain_count = parser(data_file)
    domain_count = [105, 18, 26]
    # printout = open('printout2.txt', 'w')

    NGRAMS = 2 
    BATCH_SIZE = 48
    EMBED_DIM = 128
    INIT_WEIGHT = .1


    # Big leaps vs small
    EPOCHS = 5  

    # After STEP_SIZE epochs, LEARNING_RATE * GAMMA = new LEARNING_RATE
    LEARNING_RATE = 1	# Run tests with LEARNING_RATE > 1, and LEARNING_RATE < 1
    STEP_SIZE = 1		# Run Tests with much larger STEP_SIZES 
    GAMMA = 0.99			# 0.8 - 0.9		GAMMA defaults to .1, consider adjusting running smaller GAMMA with other LR & SS

    # Calculate start time.
    start_time = time.time()

    print("NGRAMS \t BATCH_SIZE \t EMBED_DIM \t LEARNING_RATE \t STEP_SIZE \t GAMMA \n")
    print(f"{NGRAMS} \t {BATCH_SIZE} \t\t {EMBED_DIM} \t\t {LEARNING_RATE} \t\t {STEP_SIZE} \t\t {GAMMA} \n")
     	
    train_dataset, test_dataset = _setup_datasets(data=data_file, root=root, ngrams=NGRAMS, vocab=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on {device}. \n")

    # Set variables for testing and learning.
    VOCAB_SIZE = len(train_dataset.get_vocab()) # 1,308,844

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
        
        # Calculate time values.
        secs = int(time.time() - epoch_start_time)
        mins = secs / 60
        secs = secs % 60
        # if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch + 1} | time in {mins:.0f} minutes, {secs:.1f} seconds')
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        # if train_acc == 1 or valid_acc == 1:
        # 	sub_train_, sub_valid_ = traingingSplit(train_dataset, train_len)
    # Calculate time values.
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Checking the results of test dataset after %d minutes, %d seconds and %d Epochs'%(mins, secs, EPOCHS))

    test_loss, test_acc = train.test(test_dataset, BATCH_SIZE, device, model, criterion, categories)
    
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test) \n\n\n')

    # Saves the model
    torch.save(model, root+"model")

builder("BIOME-z2")