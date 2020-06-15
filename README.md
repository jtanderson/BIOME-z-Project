# BIOME-z-Project

TODO: Write project summary/notes/howto here.

## Meetings 

Prior to each meeting, make a note under each goal with your name/handle/initials and a note about what you did or worked on to address that goal

### 6/15/2020
- Next steps: integrate RDF-pulled data into text-classification framework demonstrated in [https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html).
- In addition to abstract text, use the article tags/subjects/keyword as well (remove "correct" ones before training).
- Start with "top-level" tags first.

### 6/8/2020
- When parsing RDF, sub-collection is captured by "foreign-key" style within the RDF (collections have a "hasPart" with collection id of subcollection)
- Dr. Maier will be heading the conversion to hyphenated tags within the BIOMEz library
- **Goals**
  - Upload notes, code, comments to repository for future discussion
    - D.S: Added another pytorch neural network for *convolutional* datasets - different type of dataset processing, but explores different functions in Pytorch nonetheless.
  - Continue to data imported from `.rdf` file working into python objects, capture structure. Start getting into pytorch data format.
    - D.S: Jack & I are considering only using the abstract text as input data for Pytorch.

### 6/1/2020
- Data flow: RDF format -> Python objects -> PyTorch classifier -> RDF format to organize
  - RDF parsing library: [https://rdflib.readthedocs.io/en/stable/index.html](https://rdflib.readthedocs.io/en/stable/index.html)
  - Python bibtex library: [https://bibtexparser.readthedocs.io/en/master](https://bibtexparser.readthedocs.io/en/master)
- Data labeling changes (in future): move from "Biological Distal Psychological Intermediate" to "Biological-Distal Psychological-Intermediate" tag format
- Existing library uploaded as `BIOMEz.rdf`
  - The ADMIN ONLY... folder is unclassified articles
- **Goals**
  - Get data import from `.rdf` file working into python objects
  - Begin finding data format requirements for PyTorch
