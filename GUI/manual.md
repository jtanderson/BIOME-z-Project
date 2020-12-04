### Getting started:
1. Go to the build tab and select "Edit Labels". Here you can change the labels (what categories the network will attempt to classify) to fit your classification.
2. Secondly, select the rdf file you wish to use by selecting the file in the testing tab.
3. Lastly, click the convert button below the file selection. This will convert the file into a usable format (csv).

### Training your model:
1. Once you have converted the file, go to the build tab, and select parameters. From here, you can adjust any of the parameters to your liking (see Neural Network Parameters below).
2. Click the "Build Neural Network" button and wait until the progress bar is full.
3. From here, you can see view your neural networks statistics in the statisitcs tab.
4. Additionally, you can go back to the testing tab and select a csv file using the file selector. Next, search for any key words and click on an article you wish to test.
5. Lastly, click the "<- Send" button to paste the article's title and abstract to the prediction field to the left. Click the "Predict" button to see your results.

### Searching a csv file:
Select a csv file using the file selection button in the testing tab. From there, you can search based on a title, an abstract, or both.

### Overriding / Confirming Prediction Results:
Once a prediction is made, there are options to either accept the predictions (Confirm button) or correct the prediction (Override button). Select which classification the prediction *should* fall under, then click "Override".

### Neural Network Parameters:
- Ngrams: Ngrams is how the neural network will bundle up words in a paragraph. By setting Ngrams to *two*, It will group every *two* words together in a sentence.
- Gamma: Since this neural network uses a descending learning rate, it needs to know how much to decrease per training round. The Gamma is the value the initial learning rate is multiplied by per epoch.
- Initial learning rate: The starting level the neural network starts learning at. See: [Descending Learning Rate](https://developers.google.com/machine-learning/crash-course/reducing-loss/learning-rate).
- Embedding dimension: This is how complex the neural network will be. By increasing the number, you are adding nodes to the structure.
- Epochs: This is how many times the neural network trains the model through the structure itself. Typically, you want a higher epoch.

### Development:
- Declan Sheehan: Tkinter GUI components & some subsidiary neural network Python files.
- Jack Stoetzel: Worked on much of the neural network Python files.