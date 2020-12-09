import os, sys, csv, platform, predictor, builder, torch, json
from tkinter import * # Tkinter
from tkinter import ttk, scrolledtext, filedialog, simpledialog, messagebox # Submodules
from tkintertable import TableCanvas, TableModel # Tkinter table
from PIL import ImageTk, Image as PILImage # Imaging for icon(s)
from fuzzysearch import find_near_matches # Searching csv file(s)
from tkhtmlview import HTMLLabel # Displaying html.
from markdown2 import Markdown # Converting mkdn to html.
from converter import parser # Converting .rdf to .csv & other.
from builder import stats_data
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

# Open the file contents of manual.txt instead of writing instructions
# inside of this Python file.
def importManualInfo(self):
	fd = open('./manual.md', 'r')
	self.manual_text.set(fd.read())
	fd.close()

def getDeviceType(self):
	self.type.set('Running on: ' + str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

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

	# for label in self.labelList:?????
	# 	self.labelOptions.append(label.strip())

# Opens the labels text file to update the label list.
def getLabels(self):
	fd = open('labels.txt', 'r')
	self.labelList = fd.readlines()
	fd.close() # Never forget to close your files, Thank you Dr. Park

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

# A class function used to select a file for the TESTING tab.
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

def selectFolder(self):
	temp_folder = filedialog.askdirectory(initialdir='./', title='Select a Model Folder')

	if temp_folder:
		end = temp_folder.rindex('/') + 1
		modelName = temp_folder[end:]
		start =  end - 6
		if temp_folder[start:end - 1] == '.data':
			self.CLASS_NAME = modelName
			self.wkdir.set('Current Directory: ' + self.CLASS_NAME)
			loadDefaultParameters(self, temp_folder[:end] + self.CLASS_NAME + '/')
		else:
			messagebox.showinfo('Incorrect folder',  'Please select a proper model folder.\nExample: \'./.data/example\'')

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
	else:
		_, self.CLASS_NAME = parserResults
		setDefaultParameters(self, './.data/' + self.CLASS_NAME + '/')
		self.wkdir.set('Current Directory: ' + self.CLASS_NAME)

def loadDefaultParameters(self, directory):
	pos = 0
	with open(directory + 'default-parameters.json') as json_file:
		data = json.load(json_file)
		for item in data:
			self.neuralNetworkVar[pos].set(float(data.get(item)))
			pos += 1

def setDefaultParameters(self, directory):
	JSON_FORMAT = {
		'ngrams': self.neuralNetworkVar[0].get(),
		'gamma': self.neuralNetworkVar[1].get(),
		'batch-size': self.neuralNetworkVar[2].get(),
		'initial-learn': self.neuralNetworkVar[3].get(),
		'embedding-dim': self.neuralNetworkVar[4].get(),
		'epochs': self.neuralNetworkVar[5].get()
	}

	with open(directory + 'default-parameters.json', 'w') as json_file:
		json.dump(JSON_FORMAT, json_file)

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

def prevGraph(self):
	if self.position > 0:
		showStats(self, self.position - 1)

def nextGraph(self):
	if self.position < len(self.model_stats) - 1:
		showStats(self, self.position + 1)

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


def updateToolbar(self):
	stats = self.model_stats[self.position]
	if self.position - 1 < 0:
		self.prevButton.config(state=DISABLED)
	else:
		self.prevButton.config(state=NORMAL)

	if self.position == len(self.model_stats) - 1:
		self.nextButton.config(state=DISABLED)
	else:
		self.nextButton.config(state=NORMAL)

	self.toolbarText.set('Run: ' + str(self.position + 1) + ' | Ngrams: ' + str(stats.ngram) + ' | Gamma: ' + str(stats.gamma) + ' | Batch: ' + str(stats.batch) + ' | Initial lrn rate: '+  str(stats.initlrn) + ' | Embed dim: ' + str(stats.embed) + ' | Epochs (more): ' + str(stats.epoch))
	self.generalStats.set('###### Time: ' + str(stats.time_min) + ' Minutes ' + str(stats.time_sec) + ' Second(s)\n###### Test Accuracy: ' + str(stats.test_acc) + '%\n###### Test Loss: ' + str(stats.test_loss) + '\n######Vocabulary Size: ' + str(stats.vocab_size) + '\n')
	self.genStatsLabel.set_html('<h3 style=\"text-align:center;\">Run #' + str(self.position + 1) + '</h3><br>' + self.mkdn2.convert(self.generalStats.get()))

# ===========================================================================================