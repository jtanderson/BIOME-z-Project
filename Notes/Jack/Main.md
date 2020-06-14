# Jack's Notes

## June 8 - Jun 14
	
#### Notes on RDF Triple Structures

When the data file is parsed, the data is stored in a graph made of triples. The triples have 
	1. BNode or URIRef
		- BNode: 	A string that references another object in the graph
		- URIRef: 	The source of the information, usually a URL to the document
	2. URIRef type: Gives the trype of object that is in the triple
		- Ex. Subject, link, abstract, title, surname, givenName, etc.
	3. BNode, Literal, URIRef
		- BNode:	A reference to another BNode in an associated sequence
			- Ex. A list of authors are strung togther with BNode references
		- Literal:	Exactly what it says it is, literal words/numbers
		- URIRef: 	Gives the type of object associated with the given BNode in the first entry of the triple

