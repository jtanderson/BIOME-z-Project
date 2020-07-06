# Sorry for the messy code, still working on it

import time
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, FOAF
from rdflib import Literal, URIRef, XSD

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




#file = 'tester.rdf'
file = '../../BIOME-z.rdf'
output = open("./data.csv", "w")

graph = rdflib.Graph()

start = time.time()

#Parses the .rdf files and stores it into the graph
graph.parse(file)

end = time.time()

print("{0:.2f} seconds to parse BIOME-z.rdf".format((end - start)))

# A test to see if the graph made by the .parse() function is connected (It is NOT)
# isConnected(graph)

# Some strings to look for when traversing through the graph
TITLE = "http://purl.org/dc/elements/1.1/title"
ABSTRACT = "http://purl.org/dc/terms/abstract"
SUBJECT = "http://purl.org/dc/elements/1.1/subject"

# For some reason, I think these only work as literals
BIO = Literal("biological")
PSYCH = Literal("psychological")
SOCIAL = Literal("social")

# Making a set of subject URIRef ID's. These will be used later, but we dont want repeats
collection = []

counter = -1
domain_count = [0, 0, 0]

# This will find the URIRef ID for every paper that has an abstract attached to it
start = time.time()
for s, p, o in graph:
	if ABSTRACT in p:
		collection = insert(collection, s, o, 'a')
		counter += 1
	if TITLE in p:
		collection = insert(collection, s, o, 't')
	if SUBJECT in p and BIO in o:
		collection = insert(collection, s, o, 'b')
		domain_count[0] += 1
	if SUBJECT in p and PSYCH in o:
		collection = insert(collection, s, o, 'p')
		domain_count[1] += 1
	if SUBJECT in p and SOCIAL in o:
		collection = insert(collection, s, o, 's')
		domain_count[2] += 1

end = time.time()

print("{0:.2f} seconds to ID {1:d} articles".format((end - start), counter))
print("Bio    -", domain_count[0])
print("Psych  -", domain_count[1])
print("Social -", domain_count[2])

for obj in collection:
		if obj.bio != 0:
			output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.bio), str(obj.title), str(obj.abstract)))
		if obj.psych != 0:
			output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.psych), str(obj.title), str(obj.abstract)))
		if obj.social != 0:
			output.write("\"{}\",\"{}\",\"{}\"\n".format(str(obj.social), str(obj.title), str(obj.abstract)))

output.close()

exit()