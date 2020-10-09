import os, sys, csv
from tkinter import * # Tkinter
from tkinter import ttk, scrolledtext, filedialog, simpledialog # Submodules
from tkintertable import TableCanvas, TableModel # Tkinter table
from PIL import ImageTk, Image as PILImage # Imaging for icon(s)
from fuzzysearch import find_near_matches # Searching csv file(s)
from tkhtmlview import HTMLLabel # Displaying html.
from markdown2 import Markdown # Converting mkdn to html.
from converter import parser # Converting .rdf to .csv & other.

# Author: Declan Sheehan

# TODO:
# 1: Organize the spaghetti.
# 2: Style the Gui up.
# 3: Create manual contents.


class Application(Frame):

	# Initialization: Declare important global variables, run functions to make gui components.
	def __init__(self):
		super().__init__()

		self.labelList = [] # List of labels the neural network will use to classify.
		self.getLabels()    # Read in labels from a file.

		# self.configureStyles() # Set the styling for the gui.

		# Important variables:
		self.rdf_csv_file_name, self.manual_text = StringVar(), StringVar()
		self.checkButtons = [IntVar(), IntVar()]
		self.csv_path = ''

		# Build the GUI.
		self.create_UI()

	def configureStyles(self):
		deleteme = 0
		# TODO: Create style(s) here to use in the program.

	def create_UI(self):
		# ========== Creating the base framework for the tabs ==========
		# Create a notebook to hold all four tabs.
		self.notebook = ttk.Notebook(self.master)

		# Geometrically pack the notebook to the main frame.
		self.notebook.pack(fill=BOTH, expand=YES)

		# Create four frames for the one notebook.
		self.frame_test = Frame(self.notebook, width=1000, height=1000)
		self.frame_build = Frame(self.notebook, width=1000, height=1000)
		self.frame_stats = Frame(self.notebook, width=1000, height=1000)
		self.frame_manual = Frame(self.notebook, width=1000, height=1000)

		# Edit frame parameters to fill the entire space.
		self.frame_test.pack(fill="both", expand=1)
		self.frame_build.pack(fill="both", expand=1)
		self.frame_stats.pack(fill="both", expand=1)
		self.frame_manual.pack(fill="both", expand=1)

		# Create four total tabs in the notebook with different labels.
		self.notebook.add(self.frame_test, text="Testing")
		self.notebook.add(self.frame_build, text="Building")
		self.notebook.add(self.frame_stats, text="Statistics")
		self.notebook.add(self.frame_manual, text="Manual")
		# ==============================================================

		# Run a class function to setup the testing tab.
		self.generateTestTab()
		self.generateBuildTab()
		self.generateManualTab()

# ======================================== TESTING TAB ========================================
	# Class function for creating the testing tab.
	def generateTestTab(self):
		self.rdf_csv_file_name.set("No File Chosen")

		# Separates the left half of the frame for the article testing section.
		articleTestingLF = LabelFrame(self.frame_test, text="Article Testing", height=1000, width=500)
		articleTestingLF.pack(side=LEFT, fill=BOTH, expand=YES)

		# Separates the right half of the frame for loading existing articles section.
		loadArticleLF = LabelFrame(self.frame_test, text="Load Existing Article", height=1000, width=500)
		loadArticleLF.pack(side=RIGHT, fill=BOTH, expand=YES)

		# A label to ask the user to select a file using the below button.
		chooseFileLabel = Label(loadArticleLF, text="Select an RDF file to load:")
		chooseFileLabel.place(x=5, y=20)

		# A button that opens a prompt for the user to select an rdf file to load.
		chooseRdfButton = Button(loadArticleLF, text="Choose File", command=self.openFileDialog)
		chooseRdfButton.place(x=200, y=15)

		# A label for the loaded/selected file name.
		fileNameLabel = Label(loadArticleLF, textvariable=self.rdf_csv_file_name)
		fileNameLabel.place(x=310, y=20)

		# Create an error label for invalid file types.
		self.fileError = Label(loadArticleLF, fg="red", text='Error: Invalid file format.')

		# A Button to convert an rdf file to a csv file.
		self.convertButton = Button(loadArticleLF, state=DISABLED, text='Convert to csv', command=self.convertFile)
		self.convertButton.place(x=5, y=50)

		# A label telling the user to input for a search.
		searchLabel = Label(loadArticleLF, text='Search for an Article:')
		searchLabel.place(x=5, y=103)

		# A text entry to act as the search bar.
		self.searchEntry = Entry(loadArticleLF, text='Search')
		self.searchEntry.place(relx=0.25, y=103, relwidth=0.57, height=23)

		# The search button to to execute the search.
		self.searchButton = Button(loadArticleLF, text='Search', state=DISABLED, command=self.searchCSV)
		self.searchButton.place(relx=0.840, y=103, relwidth=0.15, height=23)

		# Create a label to prompt the user to search in either title and/or abstract.
		searchInLabel = Label(loadArticleLF, text='Search in: ')
		searchInLabel.place(x=5, y=136)

		# Create a Checkbutton that will control what if the search searches in title or not.
		self.titleCB = Checkbutton(loadArticleLF, text='Title', variable=self.checkButtons[0])
		self.checkButtons[0].set(1) # Set the title check button on.
		self.titleCB.place(x=75, y=135)

		# Create a Checkbutton that will control what if the search searches in abstract or not.
		self.abstractCB = Checkbutton(loadArticleLF, text='Abstract', variable=self.checkButtons[1])
		self.abstractCB.place(x=145, y=135)

		# A lable frame for the tkintertable, since I cannot place the table anywhere in
		# the load article label frame.
		tableLF = LabelFrame(loadArticleLF)
		tableLF.place(x=3, y=165, relwidth=0.99, relheight=0.5)

		# Create the model for the table.
		self.tableModel = TableModel()

		# Set the default data.
		self.data = {num: {'Title': '', 'Abstract': ''} for num in range(25)}

		# Create the table given parameters.
		self.searchTable = TableCanvas(tableLF,
			data=self.data,
			cellwidth=325,
			model=self.tableModel,
			rowheaderwidth=0,
			showkeynamesinheader=False,
			editable=False)

		# Finally, show the table on startup.
		self.searchTable.show()

		# A button below the table to transfer the contents of the row to text fields.
		transferRowButton = Button(loadArticleLF, text='<-- Send', command=self.pushRowContents)
		transferRowButton.place(relx=0.03, rely=0.75, relwidth=0.12, height=23)
		# |---------------------------------------------------------------------|
		# Create a label to prompt the user to enter a title.
		articleTitleLabel = Label(articleTestingLF, text="Enter a title below:")
		articleTitleLabel.place(x=5, y=20)

		# Create a scrollable text for the user to input the title.
		self.titleText = scrolledtext.ScrolledText(articleTestingLF)
		self.titleText.place(x=5, y=50, relwidth=0.98, height=75)

		# Create a label to prompt the user to enter an corresponding abstract.
		abstractLabel = Label(articleTestingLF, text="Enter an abstract below:")
		abstractLabel.place(x=5, y=140)

		# Create a scrollable text for the user to input the abstract.
		self.abstractText = scrolledtext.ScrolledText(articleTestingLF)
		self.abstractText.place(x=5, y=175, relwidth=0.98, height=150)
		
		# Create a Label Frame to hold the prediction options.
		predictionsLF = LabelFrame(articleTestingLF, text='HAL 9000 predications')
		predictionsLF.place(relx=0.01, rely=0.5, relwidth=0.99, relheight=0.50)

		# A button to confirm the neural networks predictions.
		confirmButton = Button(predictionsLF, text='Confirm')
		confirmButton.place(relx=0.800, rely=0.80, relwidth=0.15, height=23)

		# An override button the user clicks in case an incorrect prediction is displayed.
		overrideButton = Button(predictionsLF, text='Override')
		overrideButton.place(relx=0.800, rely=0.90, relwidth=0.15, height=23)

		# ========== Creating an options menu for each of the labels ===========
		labelOptions = []
		for label in self.labelList:
			labelOptions.append(label.strip())

		var = StringVar(self.frame_test)
		var.set(labelOptions[0])

		labelOptionsMenu = OptionMenu(predictionsLF, var, *labelOptions)
		labelOptionsMenu.place(relx=0.02, rely=0.90, relwidth=0.2, height=23)
		# ======================================================================
# =============================================================================================



# ======================================== BUILD TAB ========================================
	# Class function to setup the building tab.
	def generateBuildTab(self):
		# Create a button to open a smaller window for label editing.
		editLabelButton = Button(self.frame_build, text='Edit Labels', command=self.openLabelWindow)
		editLabelButton.place(relx=0.80, rely=0.10, width=150, height=25)

		# Setup a button for building the network.
		buildNNButton = Button(self.frame_build, text='Build Neural Network')
		buildNNButton.place(relx=0.05, rely=0.90, width=150, height=25)

		# Setup a button for re-running the neural network.
		rerunButton = Button(self.frame_build, text='Re-run')
		rerunButton.place(relx=0.80, rely=0.90, width=150, height=25)
# ============================================================================================



# ======================================== MANUAL TAB ========================================
	# Class function to setup the manual tab.
	def generateManualTab(self):
		mkdn2 = Markdown()
		self.importManualInfo()
		# Create a title for the manual.
		manualTitleLabel = Label(self.frame_manual, text="Biome-z GUI Version 1.0", font=('Times', '25'))
		manualTitleLabel.place(x=350, y=5)

		# Create a small image at the top left corner.
		sammyImage = ImageTk.PhotoImage(PILImage.open('./sammy.ico'))
		manualIcon = Label(self.frame_manual, image=sammyImage)
		manualIcon.image = sammyImage
		manualIcon.place(x=20, y=20)

		# Make a label frame to put the HTML Label inside.
		manualLF = LabelFrame(self.frame_manual, relief=SUNKEN)
		manualLF.place(x=50, y=150, relwidth=0.900, relheight=0.700)

		# Make HTML label for the contents of the manual.md to be put in.
		manualLabel = HTMLLabel(manualLF)
		manualLabel.pack(fill=BOTH)

		# Set the contents of the manual.md to the text of the HTML Label.
		manualLabel.set_html(mkdn2.convert(self.manual_text.get()))
# ============================================================================================

	# Class function to create the smaller window for editing labels.
	def openLabelWindow(self):
		# Create the window itself.
		self.labelWindow = Toplevel(self)
		self.labelWindow.title('Edit Labels')
		self.labelWindow.geometry('400x400')

		# Create a label frame to hold the list of labels.
		listLF = LabelFrame(self.labelWindow, text='List of Labels')
		listLF.place(x=5, y=5, width=205, height=300)

		# Create a list box of the labels gathered from the labels.txt
		self.labelListBox = Listbox(listLF, bd=2, selectmode=MULTIPLE)
		self.labelListBox.pack(side=LEFT, fill=BOTH)

		# Make a scroll bar and attach it to the list box.
		labelScroll = Scrollbar(listLF)
		labelScroll.pack(side=RIGHT, fill=BOTH)
		self.labelListBox.config(yscrollcommand=labelScroll.set, font=12)
		labelScroll.config(command=self.labelListBox.yview)

		# From a class variable, insert the label in each row.
		for label in self.labelList:
			self.labelListBox.insert(END, label.strip())

		# Add a button for adding a label. It will call a corresponding function.
		addLabelButton = Button(self.labelWindow, text='Add Label', command=self.addLabel)
		addLabelButton.place(x=250, y=20, width=110, height=30)

		# Add a button for removing a label. It will also call a corresponding function.
		delLabelButton = Button(self.labelWindow, text='Remove Label', command=self.delLabel)
		delLabelButton.place(x=250, y=70, width=110, height=30)

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
			self.getLabels()
			self.updateListBox()

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
		self.getLabels()
		self.updateListBox()

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

	# Open the file contents of manual.txt instead of writing instructions
	# inside of this Python file.
	def importManualInfo(self):
		fd = open('./manual.md', 'r')
		self.manual_text.set(fd.read())
		fd.close()

	# Function to convert rdf to csv and save it under a '.data' folder.
	def convertFile(self):
		parser(self.file_path)

	# In order to let the GUI work in the back-end (pytorch/rdflib), there needs to be
	# a handful of getters and setters?
	# Get the article title the user entered.
	def getArticleTitle(self):
		return (self.titleText.get("1.0", END))

	# Get the article abstract the user entered.
	def getAbstractTitle(self):
		return (self.abstractText.get("1.0", END))		



# Define the main to start the GUI:
def main():
	# Sets the root.
	root = Tk()
	# Set the title of the Gui.
	root.title("Biomez Graphical User Interface")
	# Gives the dimensions for the program at startup.
	root.geometry("1000x1000")
	# Set the minimum size of the GUI.
	root.minsize(1000, 750)
	# Prevent resizing of the application.
	root.resizable(True, True)
	# Run the class
	app = Application()
	# Set the topbar icon for the GUI.
	topbarIcon = Image('photo', file='./sammy.ico')
	root.call('wm', 'iconphoto', root._w, topbarIcon)

	# Change the general style of the program.
	rootStyle = ttk.Style()
	rootStyle.theme_use('clam')

	# Anything after this line below will execute after the GUI is exited.
	root.mainloop()

# Run the main:
if __name__ == '__main__':
	main()
