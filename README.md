# BIOME-z-Project

TODO: Write project summary/notes/howto here.
Project team: 
 - Bevan Smith
 - Ethan Gray 

## Meetings 

Prior to each meeting, make a note under each goal with your name/handle/initials and a note about what you did or worked on to address that goal.

### 9/8/2020
- Break apart program into modules
- Try `tkinter`
- Use [PyInstaller]*https://www.pyinstaller.org/) for a "releasable" program

### 9/8/2020
- Work with APA/Psychnet for foundational model
- Break through "fast plateau"
  - more data
  - more layers/connections
- "user interface" to add and classify new articles
  - Saving/loading model
- move to CUDA/mpi backend for scalability
- review previous work on Kaggle (see Dr. Maier's notes folder for some) to get some ideas/inspiration
- **this week**
  - work on gui/frontend to add/modify labels of data
    - ask user to enter new paper or load existing one
    - present with NN "guess", allow them to accept or override with manual label
  - add the ability to re-run the training process with GUI-set parameters
    - integrate visuals/assessment/metrics for the training of the data
  
---

### 8/3/2020
- Jack will train off of the databases saved by Declan from PsycNet, then compare with models run on BIOME-z data. 
- Jack will communicate with Declan hard outcomes for Declan to share in his absence on our 8/10/20 meeting.
- Continue to develop poster ideas - probably based on 3-4 slides depicting various steps/iterative process of the project as discussed today.
- Dr. Maier continue to update & clean BIOME-z database. He will also share information about existing code to classify research articles (now posted under Notes/Karl by BPSEF).

### 7/27/2020
- Explore APA PsycNet, ResearchGate, SocArxiv, etc. as sources of extra data
- Explore journal/research articles for similar attempts, looking for additional methods to compare and contrast with this work. Think about what is the rationale for this approach, why does it make sense or improve upon existing work?
- "Interface" to tie results from model back into zotero database; automatic tagging and saving tags back to `.rdf` source 

### 7/20/2020
- When using CUDA through pytorch in hpcl, run on a workstation:
CUDA is deployed to the HPCL. I believe installing PyTorch with pip's --user flag should work, but if they run into trouble that way I can deploy it to the system image. All of the workstations have GPUs in them. hpcl1-[1-5] and hpcl[2-5]-[1-6] have workstation cards in them, and hpcl6-[1-4] have a standard graphics card in them. I haven't done any performance tests between the two cards so I don't really have any advice on which to choose.

SSH is working between hslinux and all of the cluster machines, the hostnames are:

hpcl1-1.salisbury.edu
hpcl1-2.salisbury.edu
hpcl1-3.salisbury.edu
hpcl1-4.salisbury.edu
hpcl1-5.salisbury.edu
hpcl2-1.salisbury.edu
hpcl2-2.salisbury.edu
hpcl2-3.salisbury.edu
hpcl2-4.salisbury.edu
hpcl2-5.salisbury.edu
hpcl2-6.salisbury.edu
hpcl3-1.salisbury.edu
hpcl3-2.salisbury.edu
hpcl3-3.salisbury.edu
hpcl3-4.salisbury.edu
hpcl3-5.salisbury.edu
hpcl3-6.salisbury.edu
hpcl4-1.salisbury.edu
hpcl4-2.salisbury.edu
hpcl4-3.salisbury.edu
hpcl4-4.salisbury.edu
hpcl4-5.salisbury.edu
hpcl4-6.salisbury.edu
hpcl5-1.salisbury.edu
hpcl5-2.salisbury.edu
hpcl5-3.salisbury.edu
hpcl5-4.salisbury.edu
hpcl5-5.salisbury.edu
hpcl5-6.salisbury.edu
hpcl6-1.salisbury.edu
hpcl6-2.salisbury.edu
hpcl6-3.salisbury.edu
hpcl6-4.salisbury.edu

- Maybe integrate article full text to get more data (instead of just abstract + keywords)
- Add more detailed accuracy diagnostics to see where we need more data (which categories are wrong?)
- Start working on a prototype of program to use the learned model:
  - Ask user for abstract/keyword input
  - Report predicted labels
    - Future: save back to Zotero with predicted labels

### 7/13/2020
- Work with different hyperparameters (batch size, dimensions, n-grams, etc.) to see different training/testing performance
- Start thinking about poster format and content

### 6/29/2020
- Load data with TorchText
- Experiment with different learning rates, optimizes, other hyperparameters
  - Account for different testing, training, and validation datasets
- Start data training and testing framework for BIOMEz data

### 6/15/2020
- Next steps: integrate RDF-pulled data into text-classification framework demonstrated in [https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html).
  - D.S: Created a .py file using the above tutorial. Wrote down a bunch of comments reflecting how we would setup our text classification with respect to our dataset.
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
