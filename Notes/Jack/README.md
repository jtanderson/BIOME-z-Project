Jack's Notes
=============

June 22 - June 28
------------------
### Notes on rdf_to_csv.py

#### Purpose
The program is to convert the `BIOME-z.rdf` file from Zotero to a `.csv` file that we can use to train our neural network.
The pytorch functions in the `TC Pytoch Tutorial.py` use a .csv file that contains a new article's infomation. 
Our intention is to make a `.csv` file with our data in the same format with the hopes of reusing the PyTorch Tutorial.

### Current Problems
~~1. The program does not create a connected graph from the `.rdf` file and will not read over every file when searching for tags/subjects (see program output).~~
~~2. When searching for the tags/subject, only a slect few will get identified properly. ~~
~~3. The functions used to replace `\n` characters in the `.csv` file are not catching every character.~~
~~4. The program runs in 10 - 30 seconds depending on how the graph was made and searched. ~~


June 08 - June 14
------------------	
### Notes on RDF Triple Structures

When the data file is parsed, the data is stored in a graph made of triples. The triples have 

1. Subject: BNode or URIRef
	- BNode: 	A string that references another object in the graph
	- URIRef: 	The source of the information, usually a URL to the document
2. Predicate: URIRef
	- Gives the type of object that is in the triple
		- Ex. Subject, link, abstract, title, surname, givenName, etc.
3. Object:	BNode, Literal, URIRef
	- BNode:	A reference to another BNode in an associated sequence
		- Ex. A list of authors is strung together with BNode references
	- Literal:	Exactly what it says it is, literal words/numbers
	- URIRef: 	Gives the type of object associated with the given BNode in the first entry of the triple

### Examples

The following example will use the Journal Article titled 
*Ecological and Evolutionary Forces Shaping Microbial Diversity in the Human Intestine*

#### Article in .rdf File

```xml
    <bib:Article rdf:about="http://www.sciencedirect.com/science/article/pii/S0092867406001929">
        <z:itemType>journalArticle</z:itemType>
        <dcterms:isPartOf rdf:resource="urn:issn:0092-8674"/>
        <bib:authors>
            <rdf:Seq>
                <rdf:li>
                    <foaf:Person>
                        <foaf:surname>Ley</foaf:surname>
                        <foaf:givenName>Ruth E.</foaf:givenName>
                    </foaf:Person>
                </rdf:li>
                <rdf:li>
                    <foaf:Person>
                        <foaf:surname>Peterson</foaf:surname>
                        <foaf:givenName>Daniel A.</foaf:givenName>
                    </foaf:Person>
                </rdf:li>
                <rdf:li>
                    <foaf:Person>
                        <foaf:surname>Gordon</foaf:surname>
                        <foaf:givenName>Jeffrey I.</foaf:givenName>
                    </foaf:Person>
                </rdf:li>
            </rdf:Seq>
        </bib:authors>
        <link:link rdf:resource="#item_1404"/>
        <link:link rdf:resource="#item_1405"/>
        <dc:subject>environment</dc:subject>
        <dc:subject>biological</dc:subject>
        <dc:subject>micro</dc:subject>
        <dc:subject>distal</dc:subject>
        <dc:subject>ecological</dc:subject>
        <dc:subject>diversity</dc:subject>
        <dc:title>Ecological and Evolutionary Forces Shaping Microbial Diversity in the Human Intestine</dc:title>
        <dcterms:abstract>The human gut is populated with as many as 100 trillion cells, whose collective genome, the microbiome, is a reflection of evolutionary selection pressures acting at the level of the host and at the level of the microbial cell. The ecological rules that govern the shape of microbial diversity in the gut apply to mutualists and pathogens alike.</dcterms:abstract>
        <dc:date>February 24, 2006</dc:date>
        <z:libraryCatalog>ScienceDirect</z:libraryCatalog>
        <dc:identifier>
            <dcterms:URI>
                <rdf:value>http://www.sciencedirect.com/science/article/pii/S0092867406001929</rdf:value>
            </dcterms:URI>
        </dc:identifier>
        <dcterms:dateSubmitted>2019-02-05 16:30:53</dcterms:dateSubmitted>
        <bib:pages>837-848</bib:pages>
    </bib:Article>

```

#### Article as RDFlib Triples

```
(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://purl.org/net/biblio#Article'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://www.zotero.org/namespaces/export#itemType'), rdflib.term.Literal('journalArticle'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/terms/isPartOf'), rdflib.term.URIRef('urn:issn:0092-8674'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/net/biblio#authors'), rdflib.term.BNode('Nc01daeb29d1f4de78f65ca33bdc0a0a3'))
(rdflib.term.BNode('Nc01daeb29d1f4de78f65ca33bdc0a0a3'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq'))
(rdflib.term.BNode('Nc01daeb29d1f4de78f65ca33bdc0a0a3'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#_1'), rdflib.term.BNode('N8acba8e1827140bd9e5414abb9ba87c2'))
(rdflib.term.BNode('Nc01daeb29d1f4de78f65ca33bdc0a0a3'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#_2'), rdflib.term.BNode('N548fde338f35420bacc6b5f4f3f1e696'))
(rdflib.term.BNode('Nc01daeb29d1f4de78f65ca33bdc0a0a3'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#_3'), rdflib.term.BNode('Nf157dcffedb445539577598cb8b697b3'))

(rdflib.term.BNode('N8acba8e1827140bd9e5414abb9ba87c2'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/Person'))
(rdflib.term.BNode('N8acba8e1827140bd9e5414abb9ba87c2'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/surname'), rdflib.term.Literal('Ley'))
(rdflib.term.BNode('N8acba8e1827140bd9e5414abb9ba87c2'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/givenName'), rdflib.term.Literal('Ruth E.'))

(rdflib.term.BNode('N548fde338f35420bacc6b5f4f3f1e696'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/Person'))
(rdflib.term.BNode('N548fde338f35420bacc6b5f4f3f1e696'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/surname'), rdflib.term.Literal('Peterson'))
(rdflib.term.BNode('N548fde338f35420bacc6b5f4f3f1e696'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/givenName'), rdflib.term.Literal('Daniel A.'))

(rdflib.term.BNode('Nf157dcffedb445539577598cb8b697b3'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/Person'))
(rdflib.term.BNode('Nf157dcffedb445539577598cb8b697b3'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/surname'), rdflib.term.Literal('Gordon'))
(rdflib.term.BNode('Nf157dcffedb445539577598cb8b697b3'), rdflib.term.URIRef('http://xmlns.com/foaf/0.1/givenName'), rdflib.term.Literal('Jeffrey I.'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/rss/1.0/modules/link/link'), rdflib.term.URIRef('file:///Users/jstoetzel/Programs/Research/tester.rdf#item_1404'))
(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/rss/1.0/modules/link/link'), rdflib.term.URIRef('file:///Users/jstoetzel/Programs/Research/tester.rdf#item_1405'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/subject'), rdflib.term.Literal('environment'))
(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/subject'), rdflib.term.Literal('biological'))
(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/subject'), rdflib.term.Literal('micro'))
(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/subject'), rdflib.term.Literal('distal'))
(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/subject'), rdflib.term.Literal('ecological'))
(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/subject'), rdflib.term.Literal('diversity'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/title'), rdflib.term.Literal('Ecological and Evolutionary Forces Shaping Microbial Diversity in the Human Intestine'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/terms/abstract'), rdflib.term.Literal('The human gut is populated with as many as 100 trillion cells, whose collective genome, the microbiome, is a reflection of evolutionary selection pressures acting at the level of the host and at the level of the microbial cell. The ecological rules that govern the shape of microbial diversity in the gut apply to mutualists and pathogens alike.'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/date'), rdflib.term.Literal('February 24, 2006'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://www.zotero.org/namespaces/export#libraryCatalog'), rdflib.term.Literal('ScienceDirect'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/elements/1.1/identifier'), rdflib.term.BNode('N1d85b69baaca4f74aa20340ce438d9d2'))
(rdflib.term.BNode('N1d85b69baaca4f74aa20340ce438d9d2'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://purl.org/dc/terms/URI'))
(rdflib.term.BNode('N1d85b69baaca4f74aa20340ce438d9d2'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#value'), rdflib.term.Literal('http://www.sciencedirect.com/science/article/pii/S0092867406001929'))

(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/dc/terms/dateSubmitted'), rdflib.term.Literal('2019-02-05 16:30:53'))
(rdflib.term.URIRef('http://www.sciencedirect.com/science/article/pii/S0092867406001929'), rdflib.term.URIRef('http://purl.org/net/biblio#pages'), rdflib.term.Literal('837-848'))
```
