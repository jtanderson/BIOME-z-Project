import os
from tkinter import *
from PIL import ImageTk, Image as PILImage
from tkinter import Tk, Frame, Menu, ttk, filedialog, simpledialog

# Create a class to hold the Gui:
class Application(Frame):

	def __init__(self):
		super().__init__()

		# A dynamic list of labels for the neural network.
		# This may change. It may be loaded from a file.
		self.labelList = []
		self.getLabels()

		# Execute class function to setup styles.
		# self.configureStyles()

		# A Class variable for the name of the rdf file.
		self.rdf_file_name = StringVar()

		self.manual_text = StringVar()

		self.create_UI()


	def configureStyles(self):
		deleteme = 0
		# TODO: Create style(s) here to use in the program.
		# Note: It is a pain in the butt to implement (so far).

	def create_UI(self):
		# ========== Creating the base framework for the tabs ==========
		# Create a notebook to hold all four tabs.
		self.notebook = ttk.Notebook(self.master)

		# Geometrically pack the notebook to the main frame.
		self.notebook.pack()

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
		self.generateManualTab()
		self.generateBuildTab()

		self.importManualInfo()



	def generateTestTab(self):
		self.rdf_file_name.set("No File Chosen")

		# Separates the left half of the frame for the article testing section.
		articleTestingLF = LabelFrame(self.frame_test, text="Article Testing", height=1000, width=500)
		articleTestingLF.place(x=0, y=0)

		# Separates the right half of the frame for loading existing articles section.
		loadArticleLF = LabelFrame(self.frame_test, text="Load Existing Article", height=1000, width=500)
		loadArticleLF.place(x=500, y=0)

		# A label to ask the user to select a file using the below button.
		chooseFileLabel = Label(self.frame_test, text="Select an RDF file to load:")
		chooseFileLabel.place(x=505, y=20)

		# A button that opens a prompt for the user to select an rdf file to load.
		chooseRdfButton = Button(self.frame_test, text="Choose File", command=self.openFileDialog)
		chooseRdfButton.place(x=700, y=15)

		# A label for the loaded/selected file name.
		fileNameLabel = Label(self.frame_test, textvariable=self.rdf_file_name)
		fileNameLabel.place(x=810, y=20)

		# Create an error label for invalid file types.
		self.fileError = Label(self.frame_test, fg="red", text='Error: Invalid file format.')

		# Create a label to prompt the user to enter a title.
		articleTitleLabel = Label(self.frame_test, text="Enter a title below:")
		articleTitleLabel.place(x=5,y=20)

		# Create a text field for the title of the article.
		titleText = Text(self.frame_test, height=5, width=69)
		titleText.grid(row=0, column=0, sticky='nsew', pady=40)

		# Create a scroll bar and attach it to the title text field.
		titleScroll = Scrollbar(self.frame_test, command=titleText.yview)
		titleScroll.grid(row=0, column=1, sticky='nsew', pady=40)
		titleText['yscrollcommand'] = titleScroll.set

		# Create a label to prompt the user to enter an corresponding abstract.
		abstractLabel = Label(self.frame_test, text="Enter an abstract:")
		abstractLabel.place(x=5, y=110)

		# Create a text field for the abstract of the article.
		abstractText = Text(self.frame_test, height=20, width=69)
		abstractText.grid(row=1, column=0, sticky='nsew')

		# Create another scroll bar and attach it to the abstract text field.
		abstractScroll = Scrollbar(self.frame_test, command=titleText.yview)
		abstractScroll.grid(row=1, column=1, sticky='nsew')
		abstractText['yscrollcommand'] = abstractScroll.set

		# Create a Label Frame to hold the prediction options.
		predictionsLF = LabelFrame(self.frame_test, text='HAL 9000 predications', height=550, width=490)
		predictionsLF.place(x=5, y=400)

		# A button to confirm the neural networks predictions.
		confirmButton = Button(self.frame_test, text='Confirm', height=1, width=7)
		confirmButton.place(x=400, y=800)

		# An override button the user clicks in case an incorrect prediction is displayed.
		overrideButton = Button(self.frame_test, text='Override', height=1, width=7)
		overrideButton.place(x=400, y=850)

		# ========== Creating an options menu for each of the labels ===========
		labelOptions = []
		for label in self.labelList:
			labelOptions.append(label.strip())


		var = StringVar(self.frame_test)
		var.set(labelOptions[0])

		labelOptionsMenu = OptionMenu(self.frame_test, var, *labelOptions)
		labelOptionsMenu.place(x=5, y=850)
		# ======================================================================


	# Class function to setup the manual tab.
	def generateManualTab(self):
		# Create a title for the manual.
		manualTitleLabel = Label(self.frame_manual, text="Biome-z GUI Version 1.0", font=35)
		manualTitleLabel.place(x=400, y=5)

		# Create a small image at the top left corner.
		sammyImage = ImageTk.PhotoImage(PILImage.open('./sammy.ico'))
		manualIcon = Label(self.frame_manual, image=sammyImage)
		manualIcon.image = sammyImage
		manualIcon.place(x=20, y=20)

		# Setup a label to a variable which will be populated in another function.
		manualInfoLabel = Label(self.frame_manual,
			textvariable=self.manual_text,
			bd=1, 
			relief=SUNKEN, 
			anchor='nw', 
			justify='left',
			font=15)

		manualInfoLabel.place(x=60, y=100, width=900, height=300)

	# Class function to setup the building tab.
	def generateBuildTab(self):
		# Create a button to open a smaller window for label editing.
		editLabelButton = Button(self.frame_build, text='Edit Labels', command=self.openLabelWindow)
		editLabelButton.place(x=800, y=50)

	# Class function to create the smaller window for editing labels.
	def openLabelWindow(self):
		# Create the window itself.
		self.labelWindow = Toplevel(self)
		self.labelWindow.title('Edit Labels')
		self.labelWindow.geometry('400x400')

		# Create a label frame to hold the list of labels.
		listLF = LabelFrame(self.labelWindow, text='List of Labels')
		listLF.place(x=5, y=5, width=200, height=300)

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
			fp = open('labels.txt', 'a+')
			fp.write(newLabel)
			fp.close()
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
			fp = open('labels.txt', 'w')
			for label in kept_index:
				if label == kept_index[len(kept_index) - 1]:
					pass
				else:
					label = label + '\n'
				fp.write(label)
			fp.close()

		# Get the labels from the file, and update the label list box.
		self.getLabels()
		self.updateListBox()

	# Update the label list box for the EDIT LABELS WINDOW.
	def updateListBox(self):

		# Delete "ALL" in label list box.
		self.labelListBox.delete(0, 100)

		# Add labels from the updated label list.
		for label in self.labelList:
			self.labelListBox.insert(END, label.strip())

	# Opens the labels text file to update the label list.
	def getLabels(self):
		fp = open('labels.txt', 'r')
		self.labelList = fp.readlines()
		fp.close() # Never forget to close your files, Thank you Dr. Park


	# A class function used to select a file for the TESTING tab.
	def openFileDialog(self):

		# Open up a file selection prompt for the user with two options: RDF / ALL types.
		file_path = filedialog.askopenfilename(initialdir='./', title='Select Rdf File', filetypes=(('rdf files', '*.rdf'),('all files', '*.*')))

		# If returned a file-path:
		if file_path:
			# Parse the filename and extension from the path:
			slashIndex = file_path.rindex('/') + 1
			fileName = file_path[slashIndex:]
			_, ext = os.path.splitext(fileName)

			# If the file is of incorrect format, place an error.
			if ext != '.rdf':
				self.fileError.place(x=700, y=37)
			else:
				self.fileError.place_forget()

			# Set the variable for a label in the same label.
			self.rdf_file_name.set(fileName)

	# Open the file contents of manual.txt instead of writing instructions
	# inside of this Python file.
	def importManualInfo(self):
		fp = open('./manual.txt', 'r')
		self.manual_text.set(fp.read())
		fp.close()


# Define the main to start the GUI:
def main():
	# Sets the root.
	root = Tk()
	# Set the title of the Gui.
	root.title("Biomez Graphical User Interface")
	# Gives the dimensions for the program at startup.
	root.geometry("1000x1000")
	# Prevent resizing of the application.
	root.resizable(False, False)
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
