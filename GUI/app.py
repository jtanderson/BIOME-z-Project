from builder import stats_data
from interface import *
from tkinter import *
from methods import *

class Application(Frame):
	def __init__(self):
		super().__init__()
		self.initializeVariables()
		self.configureStyles()
		getLabels(self)
		loadDefaultParameters(self, './')
		getDeviceType(self)
		create_UI(self)
	
	def initializeVariables(self):

		# Statistics variables:
		self.generalStats = StringVar()
		self.toolbarText = StringVar()
		self.model_stats = []
		self.position = 0

		self.rdf_csv_file_name, self.manual_text, self.wkdir, self.type = StringVar(), StringVar(), StringVar(), StringVar()
		self.neuralNetworkVar = [DoubleVar(), DoubleVar(), DoubleVar(), DoubleVar(), DoubleVar(), DoubleVar()]
		self.buildProgress = DoubleVar()
		self.checkButtons = [IntVar(), IntVar()]
		self.csv_path, self.CLASS_NAME = '', ''
		self.mkdn2 = Markdown()
		self.labelList, self.labelOptions = [], []

	def configureStyles(self):
		self.Program_Style = ttk.Style()
		self.Program_Style.configure('green.Horizontal.TProgressbar', foreground='green', background='black')
