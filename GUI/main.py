from app import Application
from tkinter import *

# Author: Declan Sheehan

# TODO:
# 1: Organize the spaghetti.
# 2: Style the Gui up.
# 3: Create manual contents.
# 4: Make sure that the user can get the file-name (file-name.rdf) properly:
#	- Converting rdf -> csv
#	- Think of other instances.
# 5: Make sure to close all file descriptors.
# 6: Fix opendialog file selector.
# 7: Add Confidence for Prediction.

def main():
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