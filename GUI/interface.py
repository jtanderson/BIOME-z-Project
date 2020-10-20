from tkinter import *
from tkinter import ttk
from methods import *

def create_UI(self):
	# ========== Creating the base framework for the tabs ==========
	self.notebook = ttk.Notebook(self.master)
	self.notebook.pack(fill=BOTH, expand=YES)

	self.frame_test = Frame(self.notebook)
	self.frame_build = Frame(self.notebook)
	self.frame_stats = Frame(self.notebook)
	self.frame_manual = Frame(self.notebook)

	self.frame_test.pack(fill="both", expand=1)
	self.frame_build.pack(fill="both", expand=1)
	self.frame_stats.pack(fill="both", expand=1)
	self.frame_manual.pack(fill="both", expand=1)

	self.notebook.add(self.frame_test, text="Testing")
	self.notebook.add(self.frame_build, text="Building")
	self.notebook.add(self.frame_stats, text="Statistics")
	self.notebook.add(self.frame_manual, text="Manual")
	# ==============================================================

	generateTestTab(self)
	generateBuildTab(self)
	generateManualTab(self)

# ======================================== TESTING TAB ========================================
def generateTestTab(self):
	self.rdf_csv_file_name.set("No File Chosen")

	# Separates the left half of the frame for the article testing section.
	self.articleTestingLF = LabelFrame(self.frame_test, text="Article Testing", height=1000, width=500)
	self.articleTestingLF.pack(side=LEFT, fill=BOTH, expand=YES)

	# Separates the right half of the frame for loading existing articles section.
	self.loadArticleLF = LabelFrame(self.frame_test, text="Load Existing Article", height=1000, width=500)
	self.loadArticleLF.pack(side=RIGHT, fill=BOTH, expand=YES)

	# A label to ask the user to select a file using the below button.
	self.chooseFileLabel = Label(self.loadArticleLF, text="Select an RDF file to load:")
	self.chooseFileLabel.place(x=5, y=20)

	# A button that opens a prompt for the user to select an rdf file to load.
	self.chooseRdfButton = Button(self.loadArticleLF, text="Choose File", command=lambda: openFileDialog(self))
	self.chooseRdfButton.place(x=200, y=15)

	# A label for the loaded/selected file name.
	self.fileNameLabel = Label(self.loadArticleLF, textvariable=self.rdf_csv_file_name)
	self.fileNameLabel.place(x=310, y=20)

	# Create an error label for invalid file types.
	self.fileError = Label(self.loadArticleLF, fg="red", text='Error: Invalid file format.')

	# A Button to convert an rdf file to a csv file.
	self.convertButton = Button(self.loadArticleLF, state=DISABLED, text='Convert to csv', command=lambda: convertFile(self))
	self.convertButton.place(x=5, y=50)

	# A label telling the user to input for a search.
	self.searchLabel = Label(self.loadArticleLF, text='Search for an Article:')
	self.searchLabel.place(x=5, y=103)

	# A text entry to act as the search bar.
	self.searchEntry = Entry(self.loadArticleLF, text='Search')
	self.searchEntry.place(relx=0.275, y=103, relwidth=0.50, height=23)

	# The search button to to execute the search.
	self.searchButton = Button(self.loadArticleLF, text='Search', state=DISABLED, command=lambda: searchCSV(self))
	self.searchButton.place(relx=0.840, y=103, relwidth=0.15, height=23)

	# Create a label to prompt the user to search in either title and/or abstract.
	self.searchInLabel = Label(self.loadArticleLF, text='Search in: ')
	self.searchInLabel.place(x=5, y=136)

	# Create a Checkbutton that will control what if the search searches in title or not.
	self.titleCB = Checkbutton(self.loadArticleLF, text='Title', variable=self.checkButtons[0])
	self.checkButtons[0].set(1) # Set the title check button on.
	self.titleCB.place(x=75, y=135)

	# Create a Checkbutton that will control what if the search searches in abstract or not.
	self.abstractCB = Checkbutton(self.loadArticleLF, text='Abstract', variable=self.checkButtons[1])
	self.abstractCB.place(x=145, y=135)

	# A lable frame for the tkintertable, since I cannot place the table anywhere in
	# the load article label frame.
	self.tableLF = LabelFrame(self.loadArticleLF)
	self.tableLF.place(x=3, y=165, relwidth=0.99, relheight=0.5)

	# Create the model for the table.
	self.tableModel = TableModel()

	# Set the default data.
	self.data = {num: {'Title': '', 'Abstract': ''} for num in range(25)}

	# Create the table given parameters.
	self.searchTable = TableCanvas(self.tableLF,
		data=self.data,
		cellwidth=325,
		model=self.tableModel,
		rowheaderwidth=0,
		showkeynamesinheader=False,
		editable=False
	)

	# Finally, show the table on startup.
	self.searchTable.show()

	# A button below the table to transfer the contents of the row to text fields.
	self.transferRowButton = Button(self.loadArticleLF, text='<-- Send', command=lambda: pushRowContents(self))
	self.transferRowButton.place(relx=0.03, rely=0.75, relwidth=0.12, height=23)
	# |---------------------------------------------------------------------|
	# Create a label to prompt the user to enter a title.
	self.articleTitleLabel = Label(self.articleTestingLF, text="Enter a title below:")
	self.articleTitleLabel.place(x=5, y=20)

	# Create a scrollable text for the user to input the title.
	self.titleText = scrolledtext.ScrolledText(self.articleTestingLF)
	self.titleText.place(x=5, y=50, relwidth=0.98, height=75)

	# Create a label to prompt the user to enter an corresponding abstract.
	self.abstractLabel = Label(self.articleTestingLF, text="Enter an abstract below:")
	self.abstractLabel.place(x=5, y=140)

	# Create a scrollable text for the user to input the abstract.
	self.abstractText = scrolledtext.ScrolledText(self.articleTestingLF)
	self.abstractText.place(x=5, y=175, relwidth=0.98, height=150)

	# A button to predict article results.
	self.predictionButton = Button(self.articleTestingLF, text='Predict', command=lambda: runPredictor(self))
	self.predictionButton.place(relx=0.4, rely=0.525, width=100)
	
	# Create a Label Frame to hold the prediction options.
	self.predictionsLF = LabelFrame(self.articleTestingLF, text='Predictions')
	self.predictionsLF.place(relx=0.00, rely=0.60, relwidth=1.00, relheight=0.40)

	# Create a HTML label for the prediction.
	self.predictResultLF = LabelFrame(self.predictionsLF, relief=SUNKEN)
	self.predictResultLF.place(relx=0.01, rely=0.01, relwidth=0.98, relheight=0.75)
	self.predictResultLabel = HTMLLabel(self.predictResultLF)
	self.predictResultLabel.place(relx=0.0, rely=0.00, relwidth=1.00, relheight=1.00)
	self.predictResultLabel.set_html(self.mkdn2.convert(''))

	# A button to confirm the neural networks predictions.
	self.confirmButton = Button(self.predictionsLF, text='Confirm')
	self.confirmButton.place(relx=0.800, rely=0.80, relwidth=0.15, height=23)

	# An override button the user clicks in case an incorrect prediction is displayed.
	self.overrideButton = Button(self.predictionsLF, text='Override')
	self.overrideButton.place(relx=0.800, rely=0.90, relwidth=0.15, height=23)

	# ========== Creating an options menu for each of the labels ===========
	labelOptions = []
	for label in self.labelList:
		labelOptions.append(label.strip())

	var = StringVar(self.frame_test)
	var.set(labelOptions[0])

	self.labelOptionsMenu = OptionMenu(self.predictionsLF, var, *labelOptions)
	self.labelOptionsMenu.place(relx=0.02, rely=0.90, relwidth=0.2, height=23)
	# ======================================================================
# =============================================================================================


# ======================================== BUILD TAB ========================================
# Neural Network Variables:
# NGRAMS                   [2   ->    5]
# Gamma                    [0.9 -> 0.99]
# Batch Size               [20  ->  100]
# Initial Learning Rate    [3.0 ->  6.0]
# Embedding Dimension      [32  ->  160]
# ----------------------------------------

def generateBuildTab(self):
	# Create a button to open a smaller window for label editing.
	self.editLabelButton = Button(self.frame_build, text='Edit Labels', command=lambda: openLabelWindow(self))
	self.editLabelButton.place(relx=0.80, rely=0.10, width=150, height=25)

	# Setup a button for building the network.
	self.buildNNButton = Button(self.frame_build, text='Build Neural Network', command=lambda: runBuilder(self))
	self.buildNNButton.place(relx=0.05, rely=0.90, width=150, height=25)

	# Setup a button for re-running the neural network.
	self.rerunButton = Button(self.frame_build, text='Re-run')
	self.rerunButton.place(relx=0.80, rely=0.90, width=150, height=25)

	self.ngramsScale = Scale(self.frame_build, label='NGRAMS', from_=2, to=5, tickinterval=1, orient=HORIZONTAL, variable=self.neuralNetworkVar[0])
	self.ngramsScale.place(x=50, y=150, relwidth=0.90)

	self.gammaScale = Scale(self.frame_build, label='Gamma', from_=0.90, to=0.99, tickinterval=0.01, resolution=0.01, orient=HORIZONTAL, variable=self.neuralNetworkVar[1])
	self.gammaScale.place(x=50, y=225, relwidth=0.90)

	self.batchSizeScale = Scale(self.frame_build, label='Batch Size', from_=20, to=100, tickinterval=5, orient=HORIZONTAL, variable=self.neuralNetworkVar[2])
	self.batchSizeScale.place(x=50, y=300, relwidth=0.90)

	self.initLrnRateScale = Scale(self.frame_build, label='Init Learning Rate', from_=3.0, to=6.0, tickinterval=0.2, resolution=0.01, orient=HORIZONTAL, variable=self.neuralNetworkVar[3])
	self.initLrnRateScale.place(x=50, y=375, relwidth=0.90)

	self.embedDimScale = Scale(self.frame_build, label='Embedding Dimension', from_=32, to=160, tickinterval=8, orient=HORIZONTAL, variable=self.neuralNetworkVar[4])
	self.embedDimScale.place(x=50, y=450, relwidth=0.90)

# ============================================================================================


# ======================================== MANUAL TAB ========================================
def generateManualTab(self):
	importManualInfo(self)
	# Create a title for the manual.
	self.manualTitleLabel = Label(self.frame_manual, text="Biome-z GUI Version 1.0", font=('Times', '25'))
	self.manualTitleLabel.place(x=350, y=5)

	# Create a small image at the top left corner.
	sammyImage = ImageTk.PhotoImage(PILImage.open('./sammy.ico'))
	self.manualIcon = Label(self.frame_manual, image=sammyImage)
	self.manualIcon.image = sammyImage
	self.manualIcon.place(x=20, y=20)

	# Make a label frame to put the HTML Label inside.
	self.manualLF = LabelFrame(self.frame_manual, relief=SUNKEN)
	self.manualLF.place(x=50, y=150, relwidth=0.900, relheight=0.700)

	# Make HTML label for the contents of the manual.md to be put in.
	self.manualLabel = HTMLLabel(self.manualLF)
	self.manualLabel.pack(fill=BOTH)

	# Set the contents of the manual.md to the text of the HTML Label.
	self.manualLabel.set_html(self.mkdn2.convert(self.manual_text.get()))

# ============================================================================================