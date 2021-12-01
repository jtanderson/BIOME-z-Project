# Jack Stoetzel

###################################### RDF to CSV Converter #######################################
import os
import time
import rdflib
import shutil
from shutil import copy
from rdflib import Literal, URIRef, XSD
from rdflib.namespace import RDF, RDFS, OWL, FOAF

class paper:
	def __init__(self):
		self.subject = URIRef("")
		self.title = ""
		self.abstract = ""
		self.category = 0

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
	words = words.replace('\"', '\'')
	words = words.replace('\n', " \\ ")
	return words

def insert(collection, subject, object, cat, catNum):
	marker = -1
	found = False

	for i in range(len(collection)):
		if collection[i].subject == subject:
			found = True
			marker = i

	if not found:
		obj = paper()
		obj.subject = subject
		marker = len(collection)
		collection.append(obj)
	
	if cat == 'a':
		collection[marker].abstract = replacer(str(object))
	elif cat == 't':
		collection[marker].title = replacer(str(object))
	elif cat == 'c':
		collection[marker].category = catNum

	return collection

def makeDir(rdf_file):
	name = ""
	root = "./.data/"

	if not os.path.isdir(root):
		os.mkdir(root)	
	print(rdf_file)
	for i in range(1,len(rdf_file)):
		if rdf_file[-i] == '/' and name == "":
			name = rdf_file[-(i-1):]

	newDir = root + (name.replace(".rdf", "/"))

	if not os.path.isdir(newDir):
		os.mkdir(newDir)

	if not os.path.isfile(newDir+name):
		copy(rdf_file, newDir)

	return newDir + name, newDir, name

# Converts the .rdf file to a .csv file
def parser(rdf_file, self):
	from methods import getTags

	file, newDir, name = makeDir(rdf_file)

	getTags(self, newDir)

	if os.path.exists('./labels.txt'):
		if os.path.getsize('./labels.txt'):
			shutil.copy(os.path.join(os.getcwd(),'labels.txt'), os.path.join(newDir, 'labels.txt'))
	else:
		return -1


	data_file = newDir + "data.csv"
	output = open(data_file, 'w', encoding='utf-8')

	# Unique ID for each paper
	ids_file = newDir + "ids.csv"
	ids_output = open(ids_file, 'w', encoding='utf-8')
	idsCount = 1
	

	graph = rdflib.Graph()

	start = time.time()

	graph.parse(file)

	end = time.time()

	print(f"{(end-start):.2f} seconds to parse BIOME-z.rdf \n")
	print(newDir)
	categories = []
	labels = open(newDir + 'labels.txt', 'r')

	for line in labels:
		categories.append(line.replace('\n', ''))
	categories.sort(reverse=False)
	labels.close()

	counter = -1
	domain_count = []
	true_domain_count = []

	for i in range(len(categories)):
		domain_count.append(0)
		true_domain_count.append(0)

	# A test to see if the graph made by the .parse() function is connected (It is NOT)
	# isConnected(graph)

	# Some strings to look for when traversing through the graph
	TITLE = "http://purl.org/dc/elements/1.1/title"
	ABSTRACT = "http://purl.org/dc/terms/abstract"
	SUBJECT = "http://purl.org/dc/elements/1.1/subject"

	collection = []


	# This will find the URIRef ID for every paper that has an abstract attached to it
	start = time.time()
	#TODO -- the order of this loop is very strange, change to be paper-first
	# Can not figure out inconsistency swapping loops - may have to do with graph
	for i in range(len(categories)):
		#print(f"doing category {categories[i]}")
		for s, p, o in graph:
			obj = str(o)
			obj = obj.lower()
			if ABSTRACT in p:
				collection = insert(collection, s, o, 'a', 0)
				counter += 1
			if TITLE in p:
				collection = insert(collection, s, o, 't', 0)
			if SUBJECT in p:
				# Swap outter loop here potentially 
				#print(f"'{categories[i].lower().strip()}' tested in '{obj}'")
				if categories[i].lower().strip() in obj:
					#print(f"obj = {obj}")
					true_domain_count[i] += 1
					collection = insert(collection, s, o, 'c', i+1)

	end = time.time()

	print(f"{(end-start):.2f} seconds to ID {counter} articles \n")

	newArr = true_domain_count.copy()
	newArr.sort(reverse=False)

	# Change to set sort than write to both output and ids?
	start = time.time()
	for obj in collection:
			if obj.abstract != "":
				if obj.category != 0:
					domain_count[obj.category-1] += 1
					output.write(f"\"{str(obj.category)}\",\"{str(obj.title)}\",\"{str(obj.abstract)}\"\n") #Saving this just in case
					#output.write(f"\"{str(obj.title)}\",\"{str(obj.abstract)}\"\n")	# Category is not needed
					ids_output.write(f"\"{idsCount}\",\"{str(obj.title)}\",\"{str(obj.abstract)}\"\n")
					idsCount += 1
	end = time.time()
	counter = 0

	for i in range(0, len(domain_count)):
		counter = counter + domain_count[i]
	print(f"{(end-start):.2f} seconds to write {counter} articles to a .csv file \n")

	for i in range(len(categories)):
		print(f"{categories[i]} - {domain_count[i]} / {true_domain_count[i]}")
	print()	

	output.close()
	return domain_count, name.replace('.rdf', '')
