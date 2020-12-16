import os
import time
import torch
import rdflib
from shutil import copy
from rdflib import Literal, URIRef, XSD
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from rdflib.namespace import RDF, RDFS, OWL, FOAF

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()

###############################################################
def rdfPredict(folder, rdf_file):

	# Model Setup
	root = './.data/' + folder + '/'
	model = torch.load(root + 'model')
	model.eval()

	vocab = torch.load(root + 'vocab')

	data_file = root + 'data.csv'

	labels = open(root+'labels.txt', 'r')

	categories = []
	for line in labels:
		categories.append(line.replace('\n', ''))

	labels.close()
		
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	ngram = 2
	###############################################################
	# RDF setup 

	graph = rdflib.Graph()

	start = time.time()

	graph.parse(rdf_file)

	end = time.time()

	print(f"{(end-start):.2f} seconds to parse {rdf_file} \n")


	PART = "http://purl.org/dc/terms/hasPart"	# p - This is for finding the sub-collections
	TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"	# p - For finding if the type is a collection
	COLLECTION = "http://www.zotero.org/namespaces/export#Collection"	# o - Finding if a type is a collection

	TITLE = "http://purl.org/dc/elements/1.1/title"	# p - Title of collection or paper
	ABSTRACT = "http://purl.org/dc/terms/abstract"	# p - Abstarct for a paper
	SUBJECT = "http://purl.org/dc/elements/1.1/subject"	# p - Tags on papers

	collections = []

	for c in categories:
		for s, p, o in graph:
			if TYPE in p and COLLECTION in o:
				if c in graph.value(s, URIRef(TITLE)):
					collections.append(s)
	
	unclass = "Unclassified"			
	for s, p, o in graph:
		if TYPE in p and COLLECTION in o:
			if unclass in graph.value(s, URIRef(TITLE)):
				collections.append(s)
	###############################################################
	# Predict, Label, Move
	start = time.time()

	for s, p, o in graph:
		if ABSTRACT in p:
			cat = predict(o, model, vocab, ngram)
			# print(s , "/n")
			graph.add((s, URIRef(SUBJECT), Literal(categories[cat])))
			graph.add((collections[cat],URIRef(PART), s))
			graph.remove((collections[len(collections) - 1], URIRef(PART), s))

	end = time.time()

	print(f"{(end-start):.2f} seconds to classify the papers \n")
	###############################################################

	# tmp = "new" + rdf_file
	graph.serialize(destination=rdf_file, format="xml")