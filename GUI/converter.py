# Jack Stoetzel

###################################### RDF to CSV Converter #######################################
import os
import time
import rdflib
from shutil import copy
from rdflib import Literal, URIRef, XSD
from rdflib.namespace import RDF, RDFS, OWL, FOAF

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

def makeDir(rdf_file):
	root = "./.data/"
	if not os.path.isdir(root):
		os.mkdir(root)	

	name = ""
	for i in range(1,len(rdf_file)):
		if rdf_file[-i] == '/' and name == "":
			name = rdf_file[-(i-1):]

	newDir = root + (name.replace(".rdf", "/"))
	if not os.path.isdir(newDir):
		os.mkdir(newDir)

	if not os.path.isfile(newDir+name):
		copy(rdf_file, newDir)

	return newDir + name, newDir

# Converts the .rdf file to a .csv file
def parser(rdf_file):
	
	file, newDir = makeDir(rdf_file)
	data_file = newDir + "data.csv"

	output = open(data_file, 'w', encoding='utf-8')
	# exit()

	graph = rdflib.Graph()

	start = time.time()

	#Parses the .rdf files and stores it into the graph
	graph.parse(file)

	end = time.time()

	print(f"{(end-start):.2f} seconds to parse {file}.\n")

	# A test to see if the graph made by the .parse() function is connected (It is NOT)
	# isConnected(graph)

	# Some strings to look for when traversing through the graph
	TITLE = "http://purl.org/dc/elements/1.1/title"
	ABSTRACT = "http://purl.org/dc/terms/abstract"
	SUBJECT = "http://purl.org/dc/elements/1.1/subject"

	
	BIO = "biological"
	PSYCH = "psychological"
	SOCIAL = "social"

	# Making a set of subject URIRef ID's. These will be used later, but we dont want repeats
	collection = []

	counter = -1
	true_domain_count = [0, 0, 0]
	domain_count = [0, 0, 0]

	# This will find the URIRef ID for every paper that has an abstract attached to it
	start = time.time()
	for s, p, o in graph:
		obj = str(o)
		obj = obj.lower()
		if ABSTRACT in p:
			collection = insert(collection, s, o, 'a')
			counter += 1
		if TITLE in p:
			collection = insert(collection, s, o, 't')
		if SUBJECT in p:
			if BIO in obj:
				true_domain_count[0] += 1
				collection = insert(collection, s, o, 'b')
			if PSYCH in obj:
				true_domain_count[1] += 1
				collection = insert(collection, s, o, 'p')
			if SOCIAL in obj:
				true_domain_count[2] += 1
				collection = insert(collection, s, o, 's')

	end = time.time()

	print(f"{(end-start):.2f} seconds to ID {counter} articles \n")

	start = time.time()
	for obj in collection:
			if obj.abstract != "":
				if obj.psych != 0:
					domain_count[1] += 1
					output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.psych), str(obj.title), str(obj.abstract)))
				elif obj.social != 0:
					domain_count[2] += 1
					output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.social), str(obj.title), str(obj.abstract)))
				elif obj.bio != 0:
					domain_count[0] += 1
					output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.bio), str(obj.title), str(obj.abstract)))
	
	end = time.time()
	counter = 0

	for i in range(0, len(domain_count)):
		counter = counter + domain_count[i]
	print(f"{(end-start):.2f} seconds to write {counter} articles to a .csv file \n")

	print("Bio    -", domain_count[0], '/', true_domain_count[0])
	print("Psycho  -", domain_count[1], '/', true_domain_count[1])
	print("Social -", domain_count[2], '/', true_domain_count[2])
	print()	

	output.close()
	return domain_count

