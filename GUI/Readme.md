# Biome-z Graphical User Interface

## October & November 2020

### Current status/notes:

#### Dependencies:
- `tkinter`
- `tkintertable`
- `PIL` (Pillow)
- `fuzzysearch`
- `tkhtmlview`
- `markdown2`
- `torch` (Pytorch)

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
- The build neural network buttons can now build the neural network using the set parameters.
- A progress bar for showing building progress added.
##### Statistics Tab
- Nothing
##### Manual Tab
- Added a Title, Image, and special label canvas that displays the contents of a `.md` file.
##### General
- The GUI can expand and shrink without loss of functionality.
- The GUI (depending on OS) displays SU's mascot as the topbar icon, or application icon.

#### Known issues:
- Cross-platform sizes differ (because different operating systems have different default 'styles'). This results in some buttons being too big, or not big enough.
- Currently, the Tkinter-table used for the searching the csv will have column (title) text bleed over to the adjacent column. Looks unattractive.
- If someone attempts to select multiple rows and uses `shift` or `ctrl`, some errors are thrown in the console. The program can still run, so this could be ignored.

#### To be completed:
- [X] Attach/put-together Jack Stoetzel's new parser to the GUI along with the label management system.
- [ ] Include the labels/tags column to the csv file. Increase the table row count?
- [X] Get started adding more components to the Build tab.
- [ ] Once ready, put together the statistics tab (mainly graphs and or charts).
- [ ] Fix the neural network building function(s), so it will build with a label that does not exist in rdf file.
  - For ex: "Social Studies" has 0 instances in the BIOME-z data, therefore will endlessly loop.
