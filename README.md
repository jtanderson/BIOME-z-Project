# BIOME-z-Project

## Meeting 6/1/2020

- Data flow: RDF format -> Python objects -> PyTorch classifier -> RDF format to organize
  - RDF parsing library: [https://rdflib.readthedocs.io/en/stable/index.html](https://rdflib.readthedocs.io/en/stable/index.html)
  - Python bibtex library: [https://bibtexparser.readthedocs.io/en/master](https://bibtexparser.readthedocs.io/en/master)
- Data labeling changes (in future): move from "Biological Distal Psychological Intermediate" to "Biological-Distal Psychological-Intermediate" tag format
- Existing library uploaded as `BIOMEz.rdf`
  - The ADMIN ONLY... folder is unclassified articles
- **Goals**
  - Get data import from `.rdf` file working into python objects
  - Begin finding data format requirements for PyTorch
