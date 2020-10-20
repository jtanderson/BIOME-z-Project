from tkinter import * # Tkinter
from interface import *
from methods import *

class Application(Frame):
	def __init__(self):
		super().__init__()
		self.initializeVariables()
		getLabels(self)
		create_UI(self)
	
	def initializeVariables(self):
		self.rdf_csv_file_name, self.manual_text = StringVar(), StringVar()
		self.neuralNetworkVar = [DoubleVar(), DoubleVar(), DoubleVar(), DoubleVar(), DoubleVar()]
		self.checkButtons = [IntVar(), IntVar()]
		self.csv_path, self.CLASS_NAME = '', ''
		self.mkdn2 = Markdown()
		self.labelList = []

	def configureStyles(self):
		pass # Add a style to the gui.
