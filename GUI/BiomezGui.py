from tkinter import *
from tkinter import Tk, Frame, Menu, ttk

# Create a class to hold the Gui:
class Application(Frame):

	def __init__(self):
		super().__init__()

		self.create_UI()

	def create_UI(self):
		# ========== Creating the base framework for the tabs ==========
		# Create a notebook to hold all four tabs.
		self.notebook = ttk.Notebook(self.master)

		# Geometrically pack the notebook to the main frame.
		self.notebook.pack()

		# Create four frames for the one notebook.
		self.frame_test = Frame(self.notebook, width=700, height=700)
		self.frame_build = Frame(self.notebook, width=700, height=700)
		self.frame_stats = Frame(self.notebook, width=700, height=700)
		self.frame_help = Frame(self.notebook, width=700, height=700)

		# Edit frame parameters to fill the entire space.
		self.frame_test.pack(fill="both", expand=1)
		self.frame_build.pack(fill="both", expand=1)
		self.frame_stats.pack(fill="both", expand=1)
		self.frame_help.pack(fill="both", expand=1)

		# Create four total tabs in the notebook with different labels.
		self.notebook.add(self.frame_test, text="Testing")
		self.notebook.add(self.frame_build, text="Building")
		self.notebook.add(self.frame_stats, text="Statistics")
		self.notebook.add(self.frame_help, text="Help")

		self.editHelpTab()


	def editHelpTab(self):
		butt = Button(self.frame_test, text="abcd").pack()







# Define the main to start the GUI:
def main():
	# Sets the root.
	root = Tk()
	# Set the title of the Gui.
	root.title("Biomez Graphical User Interface")
	# Gives the dimensions for the program at startup.
	root.geometry("700x700")
	# Run the class
	app = Application()
	root.mainloop()

# Run the main:
if __name__ == '__main__':
	main()