################################## Training and Validation Setup ##################################
import io
import os
import time
import torch
import logging
from tqdm import tqdm
from random import seed
from random import randint
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader

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
        f.close()

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

def _setup_datasets(data, root, ngrams=1, vocab=None, include_unk=False, rebuild=False):
	train_csv_path = root + 'train.csv'
	test_csv_path = root + 'test.csv'
	while True:
		if rebuild or (not os.path.isfile(train_csv_path) or not os.path.isfile(test_csv_path)):
			train_csv_path, test_csv_path = splitter(data, root, train_csv_path, test_csv_path)
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
		if len(train_labels ^ test_labels) == 0:
			break
		else:
			rebuild = True
			# raise ValueError("Training and test labels don't match")
	return (TextClassificationDataset(vocab, train_data, train_labels),
		TextClassificationDataset(vocab, test_data, test_labels))

def splitter(data, root, train_csv_path, test_csv_path):
	train_file = open(train_csv_path, 'w', encoding='utf-8')
	test_file = open(test_csv_path, 'w', encoding='utf-8')

	line_count = len(open(data, 'r', encoding='utf-8').readlines())
	seed(time.time())
	lines = set()

	while len(lines) < line_count * .1:
		lines.add(randint(0, line_count))

	with open(data, 'r', encoding='utf-8') as data_file:
		for lineno, line in enumerate(data_file):
			if lineno in lines:
				test_file.write(line)
			else:
				train_file.write(line)
		data_file.close()
	train_file.close()
	test_file.close()
	return train_csv_path, test_csv_path

