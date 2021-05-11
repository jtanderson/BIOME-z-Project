import os, os.path, time, sys, csv, platform, predictor, builder, torch, json

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) # For showing plots in Tk
from tkinter import ttk, scrolledtext, filedialog, simpledialog, messagebox # For Tk submodules
from tkintertable import TableCanvas, TableModel # For tkinter table in test tab.
from PIL import ImageTk, Image as PILImage # For showing images.
from fuzzysearch import find_near_matches # For searching.
from matplotlib.figure import Figure # For more MPL.
from tkhtmlview import HTMLLabel # For html in tk.
import re

# Import the statistics object dedicated to
# passing build/train data for displaying.
from builder import stats_data
from rdfPredictor import rdfPredict

from markdown2 import Markdown # For md in tk.
from converter import parser # For rdf to csv.
from tkinter import * # For more tk.

######################################## Testing Tab Functions ########################################

# This function is attached to the 'Predict' button, and will concat the abstract + title to use to
# predict, and set the output results on the prediction result html label.
def runPredictor(self):
	results = None
	if self.CLASS_NAME == '':
		messagebox.showinfo('No file selected', 'No file selected')
	elif self.abstractText.get("1.0", END) == '\n':
		messagebox.showinfo('No Abstract Set', 'Please enter an abstract.')
	else:
		options, topOpt, smValues = predictor.predictor(self.CLASS_NAME, self.titleText.get("1.0", END) + self.abstractText.get("1.0", END))
		results = ''
		for num in range(len(smValues)):
			results = results + '##### ' + options[num] + ' Confidence: ' + str(round(smValues[num], 5)) + ".\n"
		self.predictResultLabel.set_html(self.mkdn2.convert(results))

# A class function used to select a file for the testing tab.
def openFileDialog(self):
	# Open up a file selection prompt for the user with two options: RDF / ALL types.
	temp_file_path = filedialog.askopenfilename(initialdir='./', title='Select Rdf File', filetypes=(('rdf files', '*.rdf'),('all files', '*.*'),('csv files', '*.csv')))

	# If returned a file-path:
	if temp_file_path:
		self.file_path = temp_file_path
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
def searchCSV(self, event):
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
		self.classifyButton['state'] = DISABLED
	else:
		_, self.CLASS_NAME = parserResults
		setDefaultParameters(self, './.data/' + self.CLASS_NAME + '/')
		self.wkdir.set('Current Directory: ' + self.CLASS_NAME)		
		self.classifyButton['state'] = NORMAL

def selectPredictFile(self):
	file_path = filedialog.askopenfilename(initialdir='./', title='Select Rdf File', filetypes=[('rdf files', '*.rdf')])

	if not file_path:
		pass
	else:
		# Parse the filename and extension from the path:
		slashIndex = file_path.rindex('/') + 1
		fileName = file_path[slashIndex:]
		if self.CLASS_NAME != '':
			rdfPredict(self.CLASS_NAME, fileName)
		else:
			messagebox.showerror('No Model Selected', 'Please select a model in order to run')

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

#######################################################################################################


######################################## Building Tab Functions #######################################

# Gets the device type to be displayed in the build tab.
def getDeviceType(self):
	self.type.set('Running on: ' + str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

# Class function to create the smaller window for advanced settigs. -- MH
def openAdvSetWindow(self):  
	### Deactivate the 'Advanced Settings' button.
	self.AdvSetButton.config(state=DISABLED)

	### Create the window itself.
	self.settingsWindow = Toplevel(self)
	self.settingsWindow.title('Advanced Settings')
	self.settingsWindow.geometry('800x425')

	generateDefaultTab(self)
	#generateRangeTab(self)
	

	# On window exit, reactivate the button.
	def quit_settings_window():
		self.AdvSetButton.config(state=NORMAL)
		self.settingsWindow.destroy()

	self.settingsWindow.protocol('WM_DELETE_WINDOW', quit_settings_window)

# ================Default Parameters Tab
def generateDefaultTab(self):
	# Create a label frame to hold the setting categories.
	self.defaultLF = LabelFrame(self.settingsWindow, text='Set New Default Parameters')
	self.defaultLF.place(x=10, y=5, relwidth=0.98, height=400)

	#tmp_folder = os.getcwd()
	#print ("tmp_folder = ", tmp_folder)
	#if tmp_folder:
	#	end = tmp_folder.rindex('/') + 1
	#	print ("end tmp_folder = ", end)
	#	modelName = tmp_folder[end:]
	#	start =  end - 6
	#	print ("start temp_folder = ", start)
	#	print ("temp_folder[start:end - 1] = ", tmp_folder)
	#	if tmp_folder[start:end - 1] == '.data':
	#		print("tmp_folder[start:end - 1] == '.data'")
	#		#print(os.path.dirname(os.getcwd()))
	#               #gui = os.getcwd() ##C:\Users\luv2d\Documents\BIOME-z-Project\GUI
	#               #print ("curr cwd = ", os.getcwd())
	#               #os.chdir(gui)
	#               #print ("after chdir cwd = ", os.getcwd())
	#	else:
	#               print("")

	
	########## Parameters #########
	self.paramLF = LabelFrame(self.settingsWindow, text='Parameters')
	self.paramLF.place(relx=0.027, y=30, relwidth=0.95, height=315)

	self.ngramsScale = Scale(self.paramLF, label='NGRAMS', from_=2, to=8, tickinterval=1, orient=HORIZONTAL, variable=self.neuralNetworkVar[0])
	self.ngramsScale.place(relx=0.0, y=0, relwidth=0.50)

	self.gammaScale = Scale(self.paramLF, label='Gamma', from_=0.85, to=0.99, tickinterval=0.01, resolution=0.02, orient=HORIZONTAL, variable=self.neuralNetworkVar[1])
	self.gammaScale.place(relx=0.50, y=0, relwidth=0.50)

	self.batchSizeScale = Scale(self.paramLF, label='Batch Size', from_=16, to=256, tickinterval=32, orient=HORIZONTAL, variable=self.neuralNetworkVar[2])
	self.batchSizeScale.place(relx=0.0, y=75, relwidth=0.50)

	self.initLrnRateScale = Scale(self.paramLF, label='Initial Learning Rate', from_=1.0, to=7.0, tickinterval=1.00, resolution=0.01, orient=HORIZONTAL, variable=self.neuralNetworkVar[3])
	self.initLrnRateScale.place(relx=0.50, y=75, relwidth=0.50)

	self.embedDimScale = Scale(self.paramLF, label='Embedding Dimension', from_=32, to=160, tickinterval=8, orient=HORIZONTAL, variable=self.neuralNetworkVar[4])
	self.embedDimScale.place(relx=0.0, y=150, relwidth=1.00)

	self.epochLabel = Label(self.paramLF, text='Epochs:', font=('Times, 15')) 
	self.epochLabel.place(relx=0.0, y=250)

	self.epochSpin = Spinbox(self.paramLF, from_=1, to=25000000, textvariable=self.neuralNetworkVar[5], font=('Times, 15'))
	self.epochSpin.place(relx=0.1, y=252, relwidth=0.15)
	#################################

	# Creates a button to save new default parameters in the main directory.
	self.saveDefaultButton = Button(self.settingsWindow, text='Save as Default', command=lambda: setDefaultParameters(self, './'))
	self.saveDefaultButton.place(relx=0.82, y=360, width=120, height=30)#(relx=0.80, rely=0.90, relwidth=0.15)


# Class function to create the smaller window for editing labels.
def openLabelWindow(self):  
	# Variables used
	edit_Label_Font = 10
	bdSize = 2 # Border size of the lists


	# Deactivate the 'Edit Labels' button.
	self.editLabelButton.config(state=DISABLED)

	# Create the window itself.
	self.labelWindow = Toplevel(self)
	self.labelWindow.minsize(700,400)
	self.labelWindow.title('Edit Labels')
	self.labelWindow.geometry('700x400')

	# Create a label frame to hold the list of tags.
	self.listTF = LabelFrame(self.labelWindow, text='List of Tags')
	self.listTF.place(x=10, y=5, relheight=300/400, relwidth=330/700)

	# Create a label frame to hold the list of labels.
	self.listLF = LabelFrame(self.labelWindow, text='List of Labels')
	self.listLF.place(relx=360/700, rely=5/400, relwidth=330/700, relheight=300/400)

	# Create a list box of the labels gathered from the labels.txt
	labelScrollY = Scrollbar(self.listLF)
	labelScrollY.pack(side=RIGHT, fill = Y)
	labelScrollX = Scrollbar(self.listLF,orient=HORIZONTAL)
	labelScrollX.pack(side=BOTTOM, fill = X)
	self.labelListBox = Listbox(self.listLF, xscrollcommand=labelScrollX.set, yscrollcommand=labelScrollY.set, bd=bdSize, selectmode=SINGLE)
	self.labelListBox.config(font=edit_Label_Font)
	self.labelListBox.pack(side=LEFT, fill=BOTH,padx=205/700, pady=300/400, expand=True)
	labelScrollY.config(command=self.labelListBox.yview)
	labelScrollX.config(command=self.labelListBox.xview)

	# Create a list box of the labels gathered from the tagsList.txt
	tagScrollY = Scrollbar(self.listTF)
	tagScrollY.pack(side=RIGHT, fill = Y)
	tagScrollX = Scrollbar(self.listTF,orient=HORIZONTAL)
	tagScrollX.pack(side=BOTTOM, fill = X)
	self.tagListBox = Listbox(self.listTF, xscrollcommand=tagScrollX.set, yscrollcommand=tagScrollY.set, bd=bdSize, selectmode=SINGLE)
	self.tagListBox.config(font=edit_Label_Font)
	self.tagListBox.pack(side=LEFT, fill=BOTH,padx=205/700, pady=300/400, expand=True)
	tagScrollY.config(command=self.tagListBox.yview)
	tagScrollX.config(command=self.tagListBox.xview)

	# From a class variable, insert the label in each row.
	for label in self.labelList:
		self.labelListBox.insert(END, label.strip())   
		
	# From a class variable, insert the tag in each row.
	for tag in self.tagsList:
		self.tagListBox.insert(END, tag)

	# Add a button for adding a label. It will call a corresponding function.
	self.addLabelButton = Button(self.labelWindow, text='Add Label', command=lambda: addLabel(self))
	self.addLabelButton.place(relx=10/700, rely=320/400, relwidth=110/700, relheight=30/400)

	# Add a button for removing a label. It will also call a corresponding function.
	self.delLabelButton = Button(self.labelWindow, text='Remove Label', command=lambda: delLabel(self))
	self.delLabelButton.place(relx=580/700, rely=320/400, relwidth=110/700, relheight=30/400)

	# On window exit, reactivate the button.
	def quit_label_window():
		self.editLabelButton.config(state=NORMAL)
		self.labelWindow.destroy()

	self.labelWindow.protocol('WM_DELETE_WINDOW', quit_label_window)

# Class function for adding a new label.
def addLabel(self):
	newLabel_Index = self.tagListBox.curselection()

	# If the user did not enter anything: do nothing, otherwise; append to file.
	if not newLabel_Index:       
		return
	else:
		newLabel = self.tagListBox.get(newLabel_Index[0])
		fd = open('./labels.txt', 'a+')
		if os.stat("./labels.txt").st_size == 0:
			fd.write(newLabel)
		else:
			fd.write("\n"+newLabel)
		fd.close()
		labelSet(self)
		getLabels(self)
		updateListBox(self)


#  Puts all components of label.txt into a set to sort and remove redundancy 
def labelSet(self):
	fd = open('./labels.txt', 'r')
	lines = fd.readlines()
	if lines == "":
		pass
	else:    
		labels = set(lines)
		fd.close()
		labels = sorted(labels)
		fd = open('./labels.txt', 'w')
		fd.truncate(0)
		for x in labels:
			# Ignore empty cases            
			if x is "\n":
				pass              
			else:
				fd.write(x)
	fd.close()


# Function to delete labels selected.
def delLabel(self):

	# Get a tuple of the indexes selected (the ones to be deleted).
	delete_index = self.labelListBox.curselection()
	
	if len(delete_index) == 0:
		return

	# For each index in the tuple, remove it from the labels list box.
	for index in delete_index:
		self.labelListBox.delete(index, last=None)

	# Get a tuple of the REMAINING words in the label list box.
	kept_index = self.labelListBox.get(0, 100)

	# Write over the file the remaining labels (assuming there are any).
	if not kept_index:
		fd = open('labels.txt', 'w')
		fd.write("")
	else:
		fd = None
		if self.CLASS_NAME == '':
			fd = open('labels.txt', 'w')
		else:
			fd = open('labels.txt', 'w')
		for label in kept_index:
			if label == kept_index[len(kept_index) - 1]:
				pass
			else:
				label = label + "\n"
			fd.write(label)
		fd.close()
		
	# Get the labels from the file, and update the label list box.
	getLabels(self)
	updateListBox(self)

# Update the label list box for the EDIT LABELS WINDOW.
def updateListBox(self):
	# Delete "ALL" in label list box.
	self.labelListBox.delete(0, END)

	# Add labels from the updated label list and
	# the OptionMenu on the test tab.
	menu = self.labelOptionsMenu['menu']
	menu.delete(0, 'end')
	for lab in self.labelList:
		menu.add_command(label=lab,
			command=lambda value=lab: self.labelOptionVar.set(value))
		self.labelListBox.insert(END, lab.strip())

# Opens the labels text file to update the label list.
def getLabels(self):
	# Reads in tags
	if self.CLASS_NAME != '':
		getTags(self)        
		fdT = open('tagsList.txt', 'r')
		self.tagsList = fdT.readlines()
		fdT.close()
	# Case where labels.txt doesn't exist
	if os.path.exists('./labels.txt') is True:
		#open('labels.txt', 'w')
		fdL = open('labels.txt', 'r')
		self.labelList = fdL.readlines()
		fdL.close() # Never forget to close your files, Thank you Dr. Park

# A function to build the neural network. If the class name == '', then there is not model
# data selected to be built with/for.
def runBuilder(self):
	if self.CLASS_NAME == '':
		messagebox.showinfo('No Model Selected', 'Please select a model in the build tab\nor convert an rdf file.')
	else:
		self.buildProgress.set(0.0)
		stats = builder.builder(self.CLASS_NAME,
			int(self.neuralNetworkVar[0].get()),
			self.neuralNetworkVar[1].get(),
			int(self.neuralNetworkVar[2].get()),
			self.neuralNetworkVar[3].get(),
			int(self.neuralNetworkVar[4].get()),
			int(self.neuralNetworkVar[5].get()),
			self.buildProgress,
			self.master)
		if self.saveButton['state'] == DISABLED:
			self.saveButton.config(state=NORMAL)

		self.model_stats.append(stats)
		showStats(self, len(self.model_stats) - 1)

# A function to allow the user to select a model from the folder.
# May need more error checking.
def selectFolder(self):
	#--MH--disable use of advanced setting button after folder is selected
	# temporary fix for advanced settings 'set default parameter' button taking current directory not /GUI
	self.AdvSetButton.config(state=DISABLED)
	
	temp_folder = filedialog.askdirectory(initialdir='./', title='Select a Model Folder')

	if temp_folder:
		end = temp_folder.rindex('/') + 1
		modelName = temp_folder[end:]
		start =  end - 6
		if temp_folder[start:end - 1] == '.data':
			self.CLASS_NAME = modelName
			self.wkdir.set('Current Directory: ' + self.CLASS_NAME)
			os.chdir(temp_folder)
			getLabels(self)
			loadDefaultParameters(self, temp_folder[:end] + self.CLASS_NAME + '/')
			self.editLabelButton['state'] = NORMAL
			self.classifyButton['state'] = NORMAL
		else:
			messagebox.showinfo('Incorrect folder',  'Please select a proper model folder.\nExample: \'./.data/example\'')
			if self.CLASS_NAME == '':
				self.classifyButton['state'] = DISABLED
			else:
				self.classifyButton['state'] = NORMAL
	getTags(self)


# Reads the tags from the rdf file and lists them inside tagsList.txt, which will be displayed to user in
# the edit labels button to select from various exisiting tags/labels.
def getTags(self):
	# Check if tagsList.txt exisits, if not, create it within the current directory    
	if os.path.exists('./tagsList.txt') is False:
		open('tagsList.txt', 'w')    
	
	
	if self.CLASS_NAME == '':
		return
	
	# Gets rdf file path and gets modification dates of the rdf and tagsList files
	rdfRoot = './' + self.CLASS_NAME + '.rdf'
	rdfDate = time.ctime(os.path.getmtime(self.CLASS_NAME + '.rdf'))
	tagsDate = time.ctime(os.path.getmtime('tagsList.txt'))
	
	# Checks to make sure tags file is empty before filling or if the rdf has been recently updated
	if os.stat('./tagsList.txt').st_size != 0 and ((rdfDate == tagsDate) or (rdfDate < tagsDate)):       
		return 
    
	# Empty the labels file in case of any deletion of tags within the labels.txt file
	tmp = open("./labels.txt", 'w')
	tmp.truncate(0)
	tmp.close()
    
	# Reads in Tags
	tags = open(rdfRoot, 'r', encoding = 'utf-8')
	line = tags.readline() # Tmp string for reading thru rdf
	tagSet = set()    # Set for all tags
	
	# File ends with </rdf:RDF>, but with regex, it would be changed to "rdf RDF"
	# There's 3 cases where tags occur:
	# (1) <dc:subject>TagName</dc:subject>
	#
	# (2) <dc:subject>
	#          <z:AutomaticTag>
	#              <rdf:value>TagName</rdf:value>
	#          </z:AutomaticTag>
	#     </dc:subject>
	# 
	# (3) <dc:subject>
	#          <z:AutomaticTag><rdf:value>TagName</rdf:value></z:AutomaticTag>
	#     </dc:subject>
	
	while line != "rdf RDF":        
		line = regexTags(tags.readline())
		if "dc subject" in line:
			if len(line) == 10:                
				line = regexTags(tags.readline())
				if len(line) == 14: #Case (3)
					line = regexTags(tags.readline())
					tagSet.add(line[10:len(line)-10].capitalize())
					line = tags.readline()
					line = tags.readline()
				else:   # Case(2)
					tagSet.add(line[26:len(line)-27].capitalize())
					line = tags.readline()
			else:  # Case (1)             
				tagSet.add(line[11:len(line)-11].capitalize())
	tags.close()
	tagSet = sorted(tagSet)    # Sorts the set
	
	# Add Tags to label.txt
	tagFile = open('./tagsList.txt','w')
	tagFile.truncate(0)    # Empties file before writing
	for x in tagSet:
		if len(x) != 0:
			tagFile.write(x.capitalize())
			tagFile.write("\n")
	tagFile.close()


# Uses regualr expressions to clean up any useless characters and format tags
def regexTags(line):
	# Removes special characters, except '-' and ','
	tmp = re.sub('[^a-zA-Z0-9-,)(]',' ',line).strip()
	# Checks and removes anything after ',' as the tags become repetitive with little difference
	tmp = re.sub(',[\s\S]*$','',tmp)
	# Checks and removes cases of '-' being the ending char
	tmp = re.sub('[-]\Z','',tmp)
	return tmp


# Loads default parameters for a specific directory.
def loadDefaultParameters(self, directory):
	pos = 0
	with open(directory + 'default-parameters.json') as json_file:
		data = json.load(json_file)
		for item in data:
			self.neuralNetworkVar[pos].set(float(data.get(item)))
			pos += 1

# Saves default parameters for a specific directory.
def setDefaultParameters(self, directory):
	#JSON_FORMAT = {...}
	#with open(directory + 'default-parameters.json', 'w') as json_file:
		#json.dump(JSON_FORMAT, json_file)
	#-----------------------# MIKAYLA #-----------------------#
	loc = directory #'./.data/' + self.CLASS_NAME + '/'
	a_file = open(loc + 'default-parameters.json', "r")
	json_object = json.load(a_file)
	a_file.close()
	JSON_FORMAT = {
		'ngrams': self.neuralNetworkVar[0].get(),
		'gamma': self.neuralNetworkVar[1].get(),
		'batch-size': self.neuralNetworkVar[2].get(),
		'initial-learn': self.neuralNetworkVar[3].get(),
		'embedding-dim': self.neuralNetworkVar[4].get(),
		'epochs': self.neuralNetworkVar[5].get()
	}
	a_file = open(loc + 'default-parameters.json', "w")
	json.dump(JSON_FORMAT, a_file)
	a_file.close()

#######################################################################################################


######################################## Statistics Tab Functions #####################################

# Takes a position, then gets the data from the object at that position.
# Then plots the data for four different graphs.
def showStats(self, position):
	self.position = position
	stats = self.model_stats[position]
	plot_acc = self.fig_acc.add_subplot(111)
	plot_acc.cla()
	plot_acc.plot(stats.epochs, stats.train_acc)
	plot_acc.plot(stats.epochs, stats.valid_acc)
	plot_acc.legend(['Training Accuracy', 'Validation Accuracy'], loc='best')
	plot_acc.set_title('Training & Validation Accuracy')
	self.Canvas_acc.draw()

	plot_loss = self.fig_loss.add_subplot(111)
	plot_loss.cla()
	plot_loss.plot(stats.epochs, stats.train_loss)
	plot_loss.plot(stats.epochs, stats.valid_loss)
	plot_loss.legend(['Training Loss', 'Validation Loss'], loc='best')
	plot_loss.set_title('Training & Validation Loss')
	self.Canvas_loss.draw()

	pie_comp1 = self.fig_comp1.add_subplot(111)
	pie_comp1.cla()
	pie_comp1.pie(stats.train_cat_val, labels=stats.train_cat_label, autopct='%1.3f%%', shadow=True, startangle=90)
	pie_comp1.axis('equal')
	pie_comp1.set_title('Training Composition')
	self.Canvas_comp1.draw()

	pie_comp2 = self.fig_comp2.add_subplot(111)
	pie_comp2.cla()
	pie_comp2.pie(stats.test_cat_val, labels=stats.test_cat_label, autopct='%1.3f%%', shadow=True, startangle=90)
	pie_comp2.axis('equal')
	pie_comp2.set_title('Testing Composition')
	self.Canvas_comp2.draw()

	updateToolbar(self)

# Is tied to the prev button in the stats tab.
# will change the current position in the set of runs, then show the data.
def prevGraph(self):
	if self.position > 0:
		showStats(self, self.position - 1)

# Is tied to the next button in the stats tab.
# will change the current position in the set of runs, then show the data.
def nextGraph(self):
	if self.position < len(self.model_stats) - 1:
		showStats(self, self.position + 1)

# Asks the user what csv file to load from. Once loaded, adds the new 'stats' object to the list of runs.
# Will pop up a message box if the file cannot load.
def loadGraph(self):
	file_name = filedialog.askopenfilename(initialdir='./', title='Select a Csv File', filetypes=[('csv files', '*.csv')])
	if file_name and '.csv' in file_name:
		try:
			stats = stats_data()
			with open(file_name, newline='') as csvfile:
				reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
				next(reader)
				stats.ngram, stats.gamma, stats.batch, stats.initlrn, stats.embed, stats.epoch = next(reader)
				next(reader)
				for epoch, ta, va, tl, vl in reader:
					stats.epochs.append(int(epoch))
					stats.train_acc.append(float(ta))
					stats.valid_acc.append(float(va))
					stats.train_loss.append(float(tl))
					stats.valid_loss.append(float(vl))
				self.model_stats.append(stats)
				showStats(self, len(self.model_stats) - 1)
		except:
			messagebox.showinfo('Graph loading error', 'An error occurred when loading the csv file.')

# Asks the user where to save the graph location, then writes parameter data and graph data to the file.
def saveGraph(self):
	stats = self.model_stats[self.position]
	file_name = filedialog.asksaveasfile(filetypes=[('Csv', '*.csv')], defaultextension=[('Csv', '*.csv')])
	if file_name:
		writer = csv.writer(open(file_name.name, 'w', newline=''))
		writer.writerow(['Ngrams', 'Gamma', 'Batch', 'Initial Learning Rate', 'Embedding Dimension', 'Number of Epochs'])
		writer.writerow([stats.ngram, stats.gamma, stats.batch, stats.initlrn, stats.embed, stats.epoch])
		writer.writerow(['Epochs', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'])
		for num in range(len(self.model_stats[self.position].epochs)):
			writer.writerow([self.model_stats[self.position].epochs[num],
				round(self.model_stats[self.position].train_acc[num], 4),
				round(self.model_stats[self.position].valid_acc[num], 4),
				round(self.model_stats[self.position].train_loss[num], 4),
				round(self.model_stats[self.position].valid_loss[num], 4)])

# Updates the toolbar given the change in position in the set of runs.
def updateToolbar(self):
	stats = self.model_stats[self.position]

	# Disables the previous button if in the first position.
	if self.position - 1 < 0:
		self.prevButton.config(state=DISABLED)
	else:
		self.prevButton.config(state=NORMAL)

	# Disables the next button if in the last position.
	if self.position == len(self.model_stats) - 1:
		self.nextButton.config(state=DISABLED)
	else:
		self.nextButton.config(state=NORMAL)

	# Updates the toolbar text.
	self.toolbarText.set('Run: ' + str(self.position + 1) + ' | Ngrams: ' + str(stats.ngram) + ' | Gamma: ' + str(stats.gamma) + ' | Batch: ' + str(stats.batch) + ' | Initial lrn rate: '+  str(stats.initlrn) + ' | Embed dim: ' + str(stats.embed) + ' | Epochs (more): ' + str(stats.epoch))

	# Updates the general statistics label.
	self.generalStats.set('###### Time: ' + str(stats.time_min) + ' Minutes ' + str(stats.time_sec) + ' Second(s)\n###### Test Accuracy: ' + str(stats.test_acc) + '%\n###### Test Loss: ' + str(stats.test_loss) + '\n######Vocabulary Size: ' + str(stats.vocab_size) + '\n')
	self.genStatsLabel.set_html('<h3 style=\"text-align:center;\">Run #' + str(self.position + 1) + '</h3><br>' + self.mkdn2.convert(self.generalStats.get()))

#######################################################################################################


######################################## Manual Tab Functions #########################################

# Reads the content of the manual file to be displayed for the manual tab.
def importManualInfo(self):
	fd = open('./manual.md', 'r')
	self.manual_text.set(fd.read())
	fd.close()

#######################################################################################################
