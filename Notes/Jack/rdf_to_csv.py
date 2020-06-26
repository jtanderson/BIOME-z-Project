# Sorry for the messy code, still working on it

import time
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, FOAF
from rdflib import Literal, URIRef, XSD

class paper:
	def __init__(self, title, abstract, bio, psych, social):
		self.title = title
		self.abstract = abstract
		self.bio = bio
		self.psych = psych
		self.social = social

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
def replacer(title, abstract):
	title.replace("\n", "\\")
	title.replace("\"", "\"\"")
	abstract.replace("\n", "\\")
	abstract.replace("\"", "\"\"")

#file = 'tester.rdf'
file = '../../BIOME-z.rdf'
output = open("./data.csv", "w")

graph = rdflib.Graph()

#Parses the .rdf files and stores it into the graph
graph.parse(file)

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
subjects = set()

counter = 0
# domain_count = [0, 0, 0]

# This will find the URIRef ID for every paper that has an abstract attached to it
start = time.time()
for s, p, o in graph:
	if ABSTRACT in p:
		counter += 1
		subjects.add(s)
'''
	# This block of code, prints out every tuple that contains the subjects biological, psychological, and social 
	if (SUBJECT in p) and BIO in graph.value(s, p, None):
		rdfbio = (graph.value(s, p, None)).lower()
		print("BIO - ",s, " - ", p)
		print(graph.value(s, p, None), "\n")
		domain_count[0] += 1
	elif (SUBJECT in p) and PSYCH in graph.value(s, p, None):
		rdfpsych = (graph.value(s, p, None)).lower()
		print("PSYCH - ",s, " - ", p)
		print(graph.value(s, p, None), "\n")
		domain_count[1] += 1
	elif (SUBJECT in p) and SOCIAL in graph.value(s, p, None):
		rdfsocial = (graph.value(s, p, None)).lower() 
		print("SOCIAL - ", s, " - ", p)
		print(graph.value(s, p, None), "\n")
		domain_count[2] += 
'''
end = time.time()

print("{0:.2f} seconds to ID {1:d} articles".format((end - start), counter))
# print("Bio    -", domain_count[0])
# print("Psych  -", domain_count[1])
# print("Social -", domain_count[2])

# List of paper objects
list = []

domain_count = [0, 0, 0]

counter = 0

# For every URIRef ID found in the last loop, we will search the graph for the associated abstracts and tags
start = time.time()
for x in subjects:
	abstract = ""
	title = ""
	# domain = [0, 0, 0]	# Make three seperate variables
	d1 = 0
	d2 = 0
	d3 = 0
	for s, p, o in graph:
		if x == s and TITLE in p:
			title = graph.value(s, p, None)
		elif x == s and ABSTRACT in p:
			abstract = graph.value(s, p, None)
		elif x == s and SUBJECT in p:
			if BIO == graph.value(s, p, None):
				d1 = 1
				domain_count[0] += 1
			elif PSYCH == graph.value(s, p, None):
				d2 = 2
				domain_count[1] += 1
			elif SOCIAL == graph.value(s, p, None):
				d3 = 3
				domain_count[2] += 1
	# print(domain, "\n")
	# replacer(title, abstract)
	title.replace("\n", "\\")
	title.replace("\"", "\"\"")
	abstract.replace("\n", "\\")
	abstract.replace("\"", "\"\"")
	if d1 == 1:
		output.write("\"1\",\"{}\",\"{}\"\n".format(title, abstract))
		counter += 1
	elif d2 == 2:
		output.write("\"2\",\"{}\",\"{}\"\n".format(title, abstract))
		counter += 1
	elif d3 == 3:
		output.write("\"3\",\"{}\",\"{}\"\n".format(title, abstract))
		counter += 1
	# list.append(paper(title, abstract, d1, d2, d3))
end = time.time()

print("{0:.2f} seconds to get {1:d} article titles, abstracts, and BIOME domains".format(end - start, counter))
print("Bio    -", domain_count[0])
print("Psych  -", domain_count[1])
print("Social -", domain_count[2])


# For every object in the lsit that has been identified in one of the biome domains, print the paper's title and biome
# output = open("data.csv", "w")
# for obj in list:
# 		if obj.bio != 0:
# 			output.write('"' + str(obj.bio) + "\",\"" + obj.title + "\" \n")
# 		elif obj.psych != 0:
# 			output.write('"' + str(obj.psych) + "\",\"" + obj.title + "\" \n")
# 		elif obj.social != 0:
# 			output.write('"' + str(obj.social) + "\",\"" + obj.title + "\" \n")
output.close()