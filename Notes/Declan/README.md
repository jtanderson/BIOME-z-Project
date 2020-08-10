# Contents:
- Pytroch MNIST tutorial: Thoroughly commented basic neural network for MNIST dataset for people to learn/reference off of.
- Pytorch convolutional tutorial: Another tutorial in further depth with Pytorch using the Kaggle Cats & Dogs neural network.
  - Uses image manipulation library for the Cat and Dog images rather than importing pre-processed images for data.
- Pytorch text classification.py: A file containing a RNN for text classification using the AG_NEWS dataset.
- Loading Test Dataset.py: A file that is meant to load in multiple csv files "train.csv" and "valid.csv" (Using Torchtext) to make datasets for the use of our RNN (to be created later).
- Prediction_TC.py: A similar file to Jack's combined file containing loading methods from a csv file, the NN class, along with predictions that takes in a text (abstract), and outputs a prediction.
- data folder: Contains csv files for Loading Custom Datasets.
	- ./data/fulldata.csv: compiled ~330 or so biological/psychological/social titles, abstracts, and classification.
- TextRazor Parser.py: A small program that uses the TextRazor API to parse an abstract. It would be useful to parse our abstracts into a group of important words for classifying.

## More Data (Psychology and Sociology) [JULY 27, 2020]:
Since our Zotero database only had approximately 100-ish biological results and double-digit psych and social results, I exported 100 psych and social articles from PsycNet into a csv file in the hopes to train the model properly (avoid over-saturating the model with *biological* abstracts). This will hopefully allow us to observe more fitting parameters (starting lrn rate, gamma, etc) with more certainty; since there was a likelihood that it would randomly split validation/training with mostly biology examples - resulting in 100% validation. Google link to new psych and social abstracts: [Csv File](https://docs.google.com/spreadsheets/d/1ZQCQaaHQ47W3zHlk81f2ajdUicaI4dWgq3PKDNo6SW8/edit?usp=sharing)
