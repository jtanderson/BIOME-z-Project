import torch
from datasets import _setup_datasets
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator

'''
TODO:
    Print which domain of papers failed the test
    - Theory, if file is made to accurately represent data (like above) then there shoudl be frequent 9/13 due to the low number of pysch and social papers
'''

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()

def predictor(folder):
	root = './.data/' + folder + '/'
	model = torch.load(root + 'model')
	model.eval()

	data_file = root + 'data.csv'

	labels = open(root+'labels.txt', 'r')

	categories = []

	for line in labels:
		categories.append(line.replace('\n', ''))

	input_str = "There is a male predominance in autism, with a male/female ratio of 4:1 and an even higher ratio (11:1) in individuals with high functioning autism. The reasons for gender differences in ASD are unknown. Genetic and environmental factors have been implicated, but no definitive evidence exists to explain male predominance. In this review, evidence is presented to support a hypothesis that the intestinal microbiota and metabolome play a role in gender dimorphism in children with autism. Metabolic products may affect not only gastrointestinal (GI) tract and the central nervous system, but also behavior, supporting communication between GI tract and central nervous system. Furthermore, mood and anxiety may affect intestinal function, indicating bidirectional flow in the gut-brain axis. Several hormone-based hypotheses are discussed to explain the prevalence of autism in males. Observations in animal models and studies in humans on the intestinal microbiome and metabolome are reviewed to support the proposed gender dimorphism hypothesis. We hypothesize that the intestinal microbiome is a contributing factor to the prevalence of ASD in boys either directly, through microbial metabolites and/or epigenetic factors capable of regulating host gene expression through DNA methylation and/or histone modification. (PsycINFO Database Record (c) 2019 APA, all rights reserved)"	# Social

	train_dataset, test_dataset = _setup_datasets(data=data_file, root=root, ngrams=2, vocab=None, rebuild=False)

	vocab = train_dataset.get_vocab()
	model = model.to("cpu")

	return categories[predict(input_str, model, vocab, 2)]

answer = predictor('PsycNet')	# Consider a way to pass in vocab, then we can remove data_file and setup_datasets
print(f"This is a {answer} article.")