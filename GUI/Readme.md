# Biome-z Graphical User Interface

## October - 2020

### Current status/notes:

#### Dependencies:
- Basic python dep: `os`, `sys`, `csv`
- `tkinter`
- `tkintertable`
- `PIL` (Pillow)
- `fuzzysearch`
- `tkhtmlview`
- `markdown2`
- `converter` (Jack Stoetzel's rdf parsing file).

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
##### Building Tab
- Build Neural Network and Re-run buttons placed.
- Edit labels button with complete interaction (add/remove) options.
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
- [ ] Attach/put-together Jack Stoetzel's new parser to the GUI along with the label management system.
- [ ] Include the labels/tags column to the csv file. Increase the table row count?
- [ ] Get started adding more components to the Build tab.
- [ ] Once ready, put together the statistics tab (mainly graphs and or charts).
- [ ] Include a `?` as the icon image for the separate label window?
