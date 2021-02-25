# Biome-z Graphical User Interface

## October - December 2020

### Current status/notes:

#### Dependencies:
- `tkinter`
- `tkintertable`
- `PIL` (Pillow)
- `fuzzysearch`
- `tkhtmlview`
- `markdown2`
- `torch` (Pytorch)
- `matplotlib`

#### Finished components:
##### Testing Tab
- Title and Abstract text fields for article predictions
- Labels selection and confirm/override button placed for prediction purposes.
- File selection dialog button added.
- Convert-to-csv button added and implemented to convert a `.rdf` to a `.csv` file with an accompanying folder structure.
- Search entry and button for searching a selected `.csv` file.
- Two check-box buttons for searching in "Titles" or "Abstracts" or both.
- Corresponding table to display search results.
- A button to paste a selected row from the table to the article testing area.
- Predictions can now be made; it will show the label and confidence interval.
##### Building Tab
- Build Neural Network and Re-run buttons placed.
- Edit labels button with complete interaction (add/remove) options.
- Select Model button added, so the user can select a pre-existing model for re-running/testing.
- Six parameters (ngrams, gamma, batch size, initial learning rate, embedding dimension, and epochs) and their respective slider components are added for building/rerunning purposes.
- Set default parameters button allows the user to set the gui's loaded default parameters.
- Set model parameters button that allows saving parameters for a particular model.
- The build neural network buttons can now build the neural network using the set parameters.
- A progress bar for showing building progress added.
##### Statistics Tab
- A General Statistics LabelFrame to display general information about a certain run.
- A Composition LabelFrame to display pie charts on the composition of the training and testing data.
- An Accuracy / Loss LabelFrame to display two line graphs that plot the accuracy and loss accross epochs.
- A toolbar that allows the user to load run data, save run data, and navigate through previous run data.
- A text to display that particular runs parameters.
- All graphs can be saved as images, and manipulated.
##### Manual Tab
- Added a Title, Image, and special label canvas that displays the contents of a `.md` file.
##### General
- The GUI can expand and shrink without loss of functionality.
- The GUI (depending on OS) displays SU's mascot as the topbar icon, or application icon.

#### Known issues:
- Cross-platform sizes differ (because different operating systems have different default 'styles'). This results in some buttons being too big, or not big enough. This can be fixed by setting different styles for widgets.
- Currently, the Tkinter-table used for the searching the csv will have column (title) text bleed over to the adjacent column. Looks unattractive. I tried to fix this; but to no avail.
- If someone attempts to select multiple rows and uses `shift` or `ctrl`, some errors are thrown in the console. The program can still run, so this could be ignored.
- There is a lack of error handling here and there. Unsure of other known issues.

#### To be completed:
- [X] Attach/put-together Jack Stoetzel's new parser to the GUI along with the label management system.
- [X] Get started adding more components to the Build tab.
- [X] Once ready, put together the statistics tab (mainly graphs and or charts).
- [ ] Fix the neural network building function(s), so it will build with a label that does not exist in rdf file.
  - For ex: "Social Studies" has 0 instances in the BIOME-z data, therefore will endlessly loop when building.
- [ ] Add data and/or improve the NN results.
  - Consider loading premade glove vectors.
  - Parse the rdf file a little better (some html still gets passed into the NN)
  - Look into one-shot learning.
- [ ] Add functions for the Override and Confirm button on the testing tab. (Override button should 'write over' the neural network's data to correct the predicted label). (Confirm should 're-assure' the neural network is correct on its guess).
- [ ] Add a function for the Train button on the building tab. (The train button should **load** the model's data, then run through X number of epochs. Do not rebuild the NN).
- [ ] Finish adding useful information in the statistics tab.
- [ ] In the saveGraph data function, save the **Pie graph data**, and **General Run data** to the csv folder.
- [ ] Then in the loadGraph data function, load the new csv file data into the stats_data class. This will allow the Run data & Pie Charts to populate when using the `load` button on the toolbar.
- [ ] Create an executable for the GUI.
- [ ] Add time and date to run meta-data
- [ ] Move run parameters (Ngrams, etc.) to the "Run #X" content panel
