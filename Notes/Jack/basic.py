'''
Nb4bc326bbf364750bf54835988e5492e
	N876ddf43b2b549b8b07b7f567db8f541
	N70e17ca551444b74a399f07d9146bbce
	N381057324a2f482993e6ac7f0872631d
	Neaeab877917a478297d9f2daca870853
	N76399d890bef4862937c70b78d0b35f4
	http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq
	Nca01ac5653de42519b428105649d5146
	N7a83dfc536c14107861944c4ec978d28
	Ne1343c0f916e4542a34622c0fea983ff
	Nb635d933bbbc4ccb9ea856e91e8d8c30
	N05a64c7b6bfd4d5e9232319243c90f98
	N4e8841e73e414f2dafa2d3499cebd92e
	Nc737a2a6ca5b4885809a8f511b3fa9c8

(rdflib.term.BNode('N70e17ca551444b74a399f07d9146bbce

'''


'''
Jack Stoetzel
Version 1

Testing out RDFlib library.



#Note to Self: Error occurs in B-PrePPRD.rdf line 4407 item_306

 <rdf:resource rdf:resource="files/306/Stebbing et al. - 2020 - COVID-19 combining antiviral and anti-inflammator.pdf"/>
    
'''

 
import rdflib
from pprint  import pprint
from rdflib.namespace import RDF, RDFS, OWL, FOAF

file = 'tester.rdf'
#file = '../../BIOME-z.rdf'

g = rdflib.Graph()

# Prints the trype of .rdf file
print(rdflib.util.guess_format(file))

#Parses the .rdf files and stores it into the graph
g.parse(file)

# number of objects in graph
print(len(g)) 

# Prints every object in the list
# for stmt in g:
#     pprint.pprint(stmt)

# Subject, Predicate, Object
# For any subject that has an abstract, print the literal abstract
print("\n\nPrinting Abstract \n\n")
for s, p, o in g:
	if "http://purl.org/dc/terms/abstract" in p:
		print(g.value(s,p,None))


print("\n\n\n")
