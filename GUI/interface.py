from tkinter import *
from tkinter import ttk
from methods import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

# Creates the user interface.
def create_UI(self):
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

	generateTestTab(self)
	generateBuildTab(self)
	generateStatsTab(self)
	generateManualTab(self)

# ======================================== TESTING TAB ========================================
def generateTestTab(self):
	self.rdf_csv_file_name.set('No File Chosen')
	self.wkdir.set('No Current Directory.') 

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
	self.chooseRdfButton.place(x=5, y=50) #150 15

	# A label for the loaded/selected file name.
	#if (len(self.rdf_csv_file_name.get()
	#SUNKEN adds 3D effect and anchor set to "w" for west so text starts at the left
	self.fileNameLabel = Label(self.loadArticleLF, textvariable=self.rdf_csv_file_name, relief=SUNKEN, width=20, anchor="w")
	self.fileNameLabel.place(x=143, y=20) #225 20

	self.classifyButton = Button(self.loadArticleLF, state=DISABLED, text='Classify', command=lambda: selectPredictFile(self))
	self.classifyButton.place(x=85, y=50) #375 15


	# Create an error label for invalid file types.
	self.fileError = Label(self.loadArticleLF, fg="red", text='Error: Invalid file format.')

	# A Button to convert an rdf file to a csv file.
	self.convertButton = Button(self.loadArticleLF, state=DISABLED, text='Convert to csv', command=lambda: convertFile(self)) 
	self.convertButton.place(x=290, y=50) #5 50

	# Creates a label for displaying the 'working directory'.
	self.dirNameLabel = Label(self.loadArticleLF, textvariable=self.wkdir) 
	self.dirNameLabel.place(x=290, y=20) #110 4

	# A label telling the user to input for a search.
	self.searchLabel = Label(self.loadArticleLF, text='Search for an Article:')
	self.searchLabel.place(x=5, y=103)

        # Used to check if the user has already searched something
        # and to redraw the table if they search for nothing
	self.hasSearched = False
	# A text entry to act as the search bar.
	self.searchEntry = Entry(self.loadArticleLF, text='Search')
	self.searchEntry.place(relx=0.275, y=103, relwidth=0.50, height=23)
	self.searchEntry.bind('<Return>', lambda event: searchCSV(self, 0))

	# The search button to to execute the search.
	self.searchButton = Button(self.loadArticleLF, text='Search', state=DISABLED, command=lambda: searchCSV(self, self))
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
		read_only=True #editable was not a real parameter
	)

	# Finally, show the table on startup.
	self.searchTable.show()
	self.searchTable.adjustColumnWidths() #Temp fix?

        # Used to copy the current model (self.data)
	self.copyModel = ""

	# A button below the table to transfer the contents of the row to text fields.
	self.transferRowButton = Button(self.loadArticleLF, text='<-- Send', command=lambda: pushRowContents(self))
	self.transferRowButton.place(relx=0.03, rely=0.75, relwidth=0.12, height=23)

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

	# A list of the prediction results if they exist
	# This list is cleared after confirming or override the label results
	self.predictionResults = []
	
	# A button to confirm the neural networks predictions.
	# Confirm button did not work now with temp solution of savePrediction()
	self.confirmButton = Button(self.predictionsLF, text='Confirm', command=lambda: savePrediction(self))
	self.confirmButton.place(relx=0.800, rely=0.80, relwidth=0.15, height=23)

	# An override button the user clicks in case an incorrect prediction is displayed.
	# Override button did not work now with temp solution of saveOverridePrediction()
	self.overrideButton = Button(self.predictionsLF, text='Override', command=lambda: saveOverridePrediction(self))
	self.overrideButton.place(relx=0.800, rely=0.90, relwidth=0.15, height=23)

	# ========== Creating an options menu for each of the labels ===========
	self.labelOptions = []
	for label in self.labelList:
		self.labelOptions.append(label.strip())

	self.labelOptionVar = StringVar(self.frame_test)
	self.labelOptionVar.set(self.labelOptions[0])

	self.labelOptionsMenu = OptionMenu(self.predictionsLF, self.labelOptionVar, *self.labelOptions)
	self.labelOptionsMenu.place(relx=0.02, rely=0.90, relwidth=0.2, height=23)
	# ======================================================================
# =============================================================================================


# ======================================== BUILD TAB ========================================

def generateBuildTab(self):
	# Create a button to open a smaller window for label editing.
	self.editLabelButton = Button(self.frame_build, text='Edit Labels', command=lambda: openLabelWindow(self))
	self.editLabelButton.place(relx=0.80, rely=0.10, width=150, height=25)

	# Creates a label for showing the processor type used for the NN.
	self.deviceTypeLabel = Label(self.frame_build, textvariable=self.type)
	self.deviceTypeLabel.place(relx=0.80, rely=0.05, width=150, height=25)

	# Creates a label to assist user in selecting a model.
	self.modelLabel = Label(self.frame_build, text='Select a model (e.g. ./.data/modelName)')
	self.modelLabel.place(relx=0.05, rely=0.10, width=250, height=25)

	# Creates a button for selecting the model.
	self.selectFolderButton = Button(self.frame_build, text='Select Model', command=lambda: selectFolder(self))
	self.selectFolderButton.place(relx=0.30, rely=0.10)


	########## Parameter interface ##########
	self.parameterLF = LabelFrame(self.frame_build, text='Parameters')
	self.parameterLF.place(relx=0.05, y=125, relwidth=0.90, height=325)

	self.ngramsScale = Scale(self.parameterLF, label='NGRAMS', from_=2, to=8, tickinterval=1, orient=HORIZONTAL, variable=self.neuralNetworkVar[0])
	self.ngramsScale.place(relx=0.0, y=0, relwidth=0.50)

	self.gammaScale = Scale(self.parameterLF, label='Gamma', from_=0.85, to=0.99, tickinterval=0.01, resolution=0.02, orient=HORIZONTAL, variable=self.neuralNetworkVar[1])
	self.gammaScale.place(relx=0.50, y=0, relwidth=0.50)

	self.batchSizeScale = Scale(self.parameterLF, label='Batch Size', from_=16, to=256, tickinterval=32, orient=HORIZONTAL, variable=self.neuralNetworkVar[2])
	self.batchSizeScale.place(relx=0.0, y=75, relwidth=0.50)

	self.initLrnRateScale = Scale(self.parameterLF, label='Initial Learning Rate', from_=1.0, to=7.0, tickinterval=1.00, resolution=0.01, orient=HORIZONTAL, variable=self.neuralNetworkVar[3])
	self.initLrnRateScale.place(relx=0.50, y=75, relwidth=0.50)

	self.embedDimScale = Scale(self.parameterLF, label='Embedding Dimension', from_=32, to=160, tickinterval=8, orient=HORIZONTAL, variable=self.neuralNetworkVar[4])
	self.embedDimScale.place(relx=0.0, y=150, relwidth=1.00)

	self.epochLabel = Label(self.parameterLF, text='Epochs:', font=('Times, 15'))
	self.epochLabel.place(relx=0.0, y=250)

	self.epochSpin = Spinbox(self.parameterLF, from_=1, to=25000000, textvariable=self.neuralNetworkVar[5], font=('Times, 15'))
	self.epochSpin.place(relx=0.1, y=252, relwidth=0.15)
	#########################################


	# Creates a button to save parameters to the default file in the main directory.
	self.setDefaultButton = Button(self.frame_build, text='Set New Default Parameter', command=lambda: setDefaultParameters(self, './'))
	self.setDefaultButton.place(relx=0.05, rely=0.90, relwidth=0.15)

	# Setup a button for building the network from scratch.
	self.buildNNButton = Button(self.frame_build, text='Build Neural Network', command=lambda: runBuilder(self))
	self.buildNNButton.place(relx=0.30, rely=0.90, relwidth=0.15, height=25)

	# Setup a button to training the neural network for a given amount of epochs.
	# NOT YET IMPLEMENTED.
	self.trainButton = Button(self.frame_build, text='Train')
	self.trainButton.place(relx=0.55, rely=0.90, relwidth=0.15, height=25)

	# Setup a button to save parameters to the module's folder.
	# Once the module is selected and loaded, the parameters will load too.
	self.setModParamButton = Button(self.frame_build, text='Set Model Parameters', command=lambda: setDefaultParameters(self, './.data/' + self.CLASS_NAME + '/'))
	self.setModParamButton.place(relx=0.80, rely=0.90, relwidth=0.15)

	# A Progress bar to show build/training progress.
	self.progressBar = ttk.Progressbar(self.frame_build, variable=self.buildProgress, style='green.Horizontal.TProgressbar')
	self.progressBar.place(relx=0.05, rely=0.95, relwidth=0.90, height=25)

# ============================================================================================


# ======================================== STATS TAB ========================================
def generateStatsTab(self):

	###################### GENERAL ######################
	# Create a labelframe for the general statistics information.
	self.generalDataFrame = LabelFrame(self.frame_stats, text='General Data')
	self.generalDataFrame.place(relx=0.0, rely=0.0, relwidth=0.35, relheight=0.50)

	# Add an html label to display the general statistics information.
	self.genStatsLabel = HTMLLabel(self.generalDataFrame)
	self.genStatsLabel.place(x=5, y=5, relwidth=0.9725, relheight=0.9725)
	self.genStatsLabel.set_html('<h3 style=\"text-align:center;\">Run #' + str(self.position + 1) + '</h3><br>')
	#####################################################


	#################### LINE GRAPH #####################
	# Two labelframes for holding the two line graph plots.
	self.accuracyGraph = LabelFrame(self.frame_stats)
	self.accuracyGraph.place(relx=0.00, rely=0.535, relwidth=0.50, relheight=0.465)
	self.lossGraph = LabelFrame(self.frame_stats)
	self.lossGraph.place(relx=0.50, rely=0.535, relwidth=0.50, relheight=0.465)

	# Create the special Tkcanvas for showing matplotlib graphs for accuracy and loss.
	self.fig_acc = Figure(figsize=(7, 5), dpi=100)
	self.Canvas_acc = FigureCanvasTkAgg(self.fig_acc, self.accuracyGraph)
	self.toolbar_acc = NavigationToolbar2Tk(self.Canvas_acc, self.accuracyGraph)
	self.toolbar_acc.update()
	self.Canvas_acc.get_tk_widget().place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)

	self.fig_loss = Figure(figsize=(7, 5), dpi=100)
	self.Canvas_loss = FigureCanvasTkAgg(self.fig_loss, self.lossGraph)
	self.toolbar_loss = NavigationToolbar2Tk(self.Canvas_loss, self.lossGraph)
	self.toolbar_loss.update()
	self.Canvas_loss.get_tk_widget().place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
	#####################################################


	#################### COMPOSITION ####################
	# Create a labelframe for the pie charts (data composition).
	self.compositionFrame = LabelFrame(self.frame_stats, text='Composition')
	self.compositionFrame.place(relx=0.35, rely=0.0, relwidth=0.65, relheight=0.50)

	# Two labelframes for each pie chart.
	self.trainLF = LabelFrame(self.compositionFrame)
	self.trainLF.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)
	self.testLF = LabelFrame(self.compositionFrame)
	self.testLF.place(relx=0.5, rely=0.0, relwidth=0.5, relheight=1.0)

	self.fig_comp1 = Figure(figsize=(1, 1))
	self.Canvas_comp1 = FigureCanvasTkAgg(self.fig_comp1, self.trainLF)
	self.toolbar_comp1 = NavigationToolbar2Tk(self.Canvas_comp1, self.trainLF)
	self.toolbar_comp1.update()
	self.Canvas_comp1.get_tk_widget().place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)

	self.fig_comp2 = Figure(figsize=(4, 4))
	self.Canvas_comp2 = FigureCanvasTkAgg(self.fig_comp2, self.testLF)
	self.toolbar_comp2 = NavigationToolbar2Tk(self.Canvas_comp2, self.testLF)
	self.toolbar_comp2.update()
	self.Canvas_comp2.get_tk_widget().place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
	#####################################################


	###################### TOOLBAR ######################
	# Create a labelframe to hold the toolbar buttons and label.
	self.toolbar = LabelFrame(self.frame_stats)
	self.toolbar.place(relx=0.0, rely=0.50, relwidth=1.00, relheight=0.035)

	self.loadButton = Button(self.toolbar, text='Load', command = lambda: loadGraph(self))
	self.loadButton.place(relx=0.0, rely=0.0, relwidth=0.0875, relheight=1.00)

	self.saveButton = Button(self.toolbar, text='Save', command = lambda: saveGraph(self), state=DISABLED)
	self.saveButton.place(relx=0.0875, rely=0.0, relwidth=0.0875, relheight=1.00)

	self.prevButton = Button(self.toolbar, text='Prev', command = lambda: prevGraph(self), state=DISABLED)
	self.prevButton.place(relx=0.175, rely=0.0, relwidth=0.0875, relheight=1.00)

	self.nextButton = Button(self.toolbar, text='Next', command = lambda: nextGraph(self), state=DISABLED)
	self.nextButton.place(relx=0.2625, rely=0.0, relwidth=0.0875, relheight=1.00)

	self.toolbarText.set('Run: 0 | Ngrams: 0 | Gamma: 0 | Batch: 0 | Initial lrn rate: 0 | Embed dim: 0 | Epochs (more): 0')
	self.toolbarParams = Label(self.toolbar, textvariable=self.toolbarText, anchor='w', font=('', '9', 'bold'))
	self.toolbarParams.place(relx=0.3525, rely=0.0, relwidth=0.5975, relheight=1.00)
	#####################################################

# ============================================================================================


# ======================================== MANUAL TAB ========================================
def generateManualTab(self):
	importManualInfo(self)
	# Create a title for the manual.
	self.manualTitleLabel = Label(self.frame_manual, text="Biome-z GUI Version 1.0", font=('Times', '25'))
	self.manualTitleLabel.place(relx=0.375, y=5)

	# Make a label frame to put the HTML Label inside.
	self.manualLF = LabelFrame(self.frame_manual, relief=SUNKEN)
	self.manualLF.place(relx=0.025, y=150, relwidth=0.95, relheight=0.75)

	# Make HTML label for the contents of the manual.md to be put in.
	self.manualLabel = HTMLLabel(self.manualLF)
	self.manualLabel.pack(fill=BOTH, expand=True)

	# Set the contents of the manual.md to the text of the HTML Label.
	self.manualLabel.set_html(self.mkdn2.convert(self.manual_text.get()))

# ============================================================================================
