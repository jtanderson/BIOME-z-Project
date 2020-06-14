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

# file = 'tester.rdf'
file = '../../BIOME-z.rdf'

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

print("--- printing raw triples ---") 
for s, p, o in g:
	print((s, p, o))


print("\n\n\n")
