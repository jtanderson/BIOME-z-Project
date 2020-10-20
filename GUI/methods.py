import os, sys, csv, platform, predictor, builder
from tkinter import * # Tkinter
from tkinter import ttk, scrolledtext, filedialog, simpledialog, messagebox # Submodules
from tkintertable import TableCanvas, TableModel # Tkinter table
from PIL import ImageTk, Image as PILImage # Imaging for icon(s)
from fuzzysearch import find_near_matches # Searching csv file(s)
from tkhtmlview import HTMLLabel # Displaying html.
from markdown2 import Markdown # Converting mkdn to html.
from converter import parser # Converting .rdf to .csv & other.

# Open the file contents of manual.txt instead of writing instructions
# inside of this Python file.
def importManualInfo(self):
	fd = open('./manual.md', 'r')
	self.manual_text.set(fd.read())
	fd.close()

# Class function to create the smaller window for editing labels.
def openLabelWindow(self):
	# Create the window itself.
	self.labelWindow = Toplevel(self)
	self.labelWindow.title('Edit Labels')
	self.labelWindow.geometry('400x400')

	# Create a label frame to hold the list of labels.
	self.listLF = LabelFrame(self.labelWindow, text='List of Labels')
	self.listLF.place(x=5, y=5, width=205, height=300)

	# Create a list box of the labels gathered from the labels.txt
	self.labelListBox = Listbox(self.listLF, bd=2, selectmode=MULTIPLE)
	self.labelListBox.pack(side=LEFT, fill=BOTH)

	# Make a scroll bar and attach it to the list box.
	self.labelScroll = Scrollbar(self.listLF)
	self.labelScroll.pack(side=RIGHT, fill=BOTH)
	self.labelListBox.config(yscrollcommand=self.labelScroll.set, font=12)
	self.labelScroll.config(command=self.labelListBox.yview)

	# From a class variable, insert the label in each row.
	for label in self.labelList:
		self.labelListBox.insert(END, label.strip())

	# Add a button for adding a label. It will call a corresponding function.
	self.addLabelButton = Button(self.labelWindow, text='Add Label', command=lambda: addLabel(self))
	self.addLabelButton.place(x=250, y=20, width=110, height=30)

	# Add a button for removing a label. It will also call a corresponding function.
	self.delLabelButton = Button(self.labelWindow, text='Remove Label', command=lambda: delLabel(self))
	self.delLabelButton.place(x=250, y=70, width=110, height=30)


# ================================= CLASS FUNCTIONS ========================================
# Class function for adding a new label.
def addLabel(self):
	# Execute an input dialog for the user to add a new label.
	newLabel = simpledialog.askstring('Input', 'Enter a new label:',
		parent=self.labelWindow)

	# Remove extrameous spaghetti the user inputted.
	if newLabel:
		newLabel = newLabel.strip()

	# If the user did not enter anything: do nothing, otherwise; append to file.
	if newLabel is None:
		return
	else:
		newLabel = '\n' + newLabel
		fd = open('labels.txt', 'a+')
		fd.write(newLabel)
		fd.close()
		getLabels(self)
		updateListBox(self)

# Function to delete labels selected.
def delLabel(self):

	# Get a tuple of the indexes selected (the ones to be deleted).
	delete_index = self.labelListBox.curselection()

	# For each index in the tuple, remove it from the labels list box.
	for index in delete_index:
		self.labelListBox.delete(index, last=None)

	# Get a tuple of the REMAINING words in the label list box.
	kept_index = self.labelListBox.get(0, 100)

	# Write over the file the remaining labels (assuming there are any).
	if not kept_index:
		pass
	else:
		fd = open('labels.txt', 'w')
		for label in kept_index:
			if label == kept_index[len(kept_index) - 1]:
				pass
			else:
				label = label + '\n'
			fd.write(label)
		fd.close()

	# Get the labels from the file, and update the label list box.
	getLabels(self)
	updateListBox(self)


# Update the label list box for the EDIT LABELS WINDOW.
def updateListBox(self):

	# Delete "ALL" in label list box.
	self.labelListBox.delete(0, END)

	# Add labels from the updated label list.
	for label in self.labelList:
		self.labelListBox.insert(END, label.strip())

# Opens the labels text file to update the label list.
def getLabels(self):
	fd = open('labels.txt', 'r')
	self.labelList = fd.readlines()
	fd.close() # Never forget to close your files, Thank you Dr. Park


def runPredictor(self):
	results = None
	if self.CLASS_NAME == '':
		messagebox.showinfo('No file selected', 'No file selected')
	elif not self.abstractText.get("1.0", END): # Likely does not work. fix.
		messagebox.showinfo('No Abstract Set', 'Please enter an abstract.')
	else:
		results = predictor.predictor(self.CLASS_NAME, self.abstractText.get("1.0", END))
		self.predictResultLabel.set_html(self.mkdn2.convert(results))

def runBuilder(self):
	if self.CLASS_NAME == '':
		messagebox.showinfo('File Needed!', 'File Needed...')
	else:
		builder.builder(self.CLASS_NAME,
			int(self.neuralNetworkVar[0].get()),
			self.neuralNetworkVar[1].get(),
			int(self.neuralNetworkVar[2].get()),
			self.neuralNetworkVar[3].get(),
			int(self.neuralNetworkVar[4].get()))

# A class function used to select a file for the TESTING tab.
def openFileDialog(self):

	# Open up a file selection prompt for the user with two options: RDF / ALL types.
	self.file_path = filedialog.askopenfilename(initialdir='./', title='Select Rdf File', filetypes=(('rdf files', '*.rdf'),('all files', '*.*'),('csv files', '*.csv')))

	# If returned a file-path:
	if self.file_path:
		# Parse the filename and extension from the path:
		slashIndex = self.file_path.rindex('/') + 1
		fileName = self.file_path[slashIndex:]
		_, ext = os.path.splitext(fileName)

		# If the file is of incorrect format, place an error.
		if ext != '.rdf' and ext != '.csv':
			self.fileError.place(x=200, y=40)
			self.convertButton['state'] = DISABLED
			self.searchButton['state'] = DISABLED
		elif ext == '.rdf':
			self.convertButton['state'] = ACTIVE
			self.fileError.place_forget()
			self.searchButton['state'] = DISABLED
		elif ext == '.csv':
			self.csv_path = self.file_path
			self.searchButton['state'] = ACTIVE
			self.convertButton['state'] = DISABLED
			self.fileError.place_forget()

		# Set the variable for a label in the same label.
		self.rdf_csv_file_name.set(fileName)


# A function tied to the search button that queries and displays results in the table.
def searchCSV(self):
	# Clear the table.
	for num in range(25):
		self.searchTable.model.deleteCellRecord(num, 0)
		self.searchTable.model.deleteCellRecord(num, 1)
	# Get the search entry.
	find = self.searchEntry.get()
	count = 0
	# Get each row of the csv file:
	csv_file = csv.reader(open(self.csv_path, 'r', encoding='utf-8'), delimiter=',')
	
	# Loop through to see if the input gets any matches, then display them in to table.
	for row in csv_file:
		if count > 24:
			break
		if self.checkButtons[0].get() == 1 and self.checkButtons[1].get() == 0 or self.checkButtons[0].get() == 0 and self.checkButtons[1].get() == 0:
			result = find_near_matches(find, row[1], max_deletions=1, max_insertions=1, max_substitutions=0)
		elif self.checkButtons[0].get() == 0 and self.checkButtons[1].get() == 1:				
			result = find_near_matches(find, row[2], max_deletions=1, max_insertions=1, max_substitutions=0)
		else:
			both = row[1] + ' ' + row[2]
			result = find_near_matches(find, both, max_deletions=1, max_insertions=1, max_substitutions=0)
		if not not result:
			self.searchTable.model.setValueAt(row[1], count, 0)
			self.searchTable.model.setValueAt(row[2], count, 1)
			count += 1

	# Update the table.
	self.searchTable.redrawTable()

# Function to convert rdf to csv and save it under a '.data' folder.
def convertFile(self):
	parserResults = parser(self.file_path)
	if parserResults == -1:
		messagebox.showerror('No Labels', 'Please add your label(s) in the build tab.')
	else:
		_, self.CLASS_NAME = parserResults

# This function grabs the contents of the table's row and implants the contents
# in the title and abstract text fields.
def pushRowContents(self):
	# Delete text field entry
	self.titleText.delete("1.0", END)
	self.abstractText.delete("1.0", END)
	# Get the selected row.
	row = self.searchTable.getSelectedRow()
	# Insert text for title and abstract.
	self.titleText.insert(INSERT, self.searchTable.model.getValueAt(row, 0))
	self.abstractText.insert(INSERT, self.searchTable.model.getValueAt(row, 1))
# ===========================================================================================