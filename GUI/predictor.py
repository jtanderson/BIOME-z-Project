import torch
from datasets import _setup_datasets
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator

'''
TODO:
    Print which domain of papers failed the test
    - Theory, if file is made to accurately represent data (like above) then there should be frequent 9/13 due to the low number of pysch and social papers
'''

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()

def predictor(folder, input_str):
	root = './.data/' + folder + '/'
	model = torch.load(root + 'model')
	model.eval()
	data_file = root + 'data.csv'

	labels = open(root+'labels.txt', 'r')

	categories = []

	for line in labels:
		categories.append(line.replace('\n', ''))

	train_dataset, test_dataset = _setup_datasets(data=data_file, root=root, ngrams=2, vocab=None, rebuild=False)

	vocab = train_dataset.get_vocab()

	model = model.to('cpu')

	return categories[predict(input_str, model, vocab, 2)]

# answer = predictor('PsycNet')	# Consider a way to pass in vocab, then we can remove data_file and setup_datasets
# print(f"This is a {answer} article.")