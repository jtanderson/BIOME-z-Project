from app import Application
from tkinter import *

# Author: Declan Sheehan, Jack Stoetzel

# TODO (If you so choose to):
# 1: Style the Gui up.
# Create different styles in configureStyles in app.py
# apply them to different widgets using style attribute.
# Not all widgets have this attr, which is sad panda.

# 2: Fix training split function in classification.py
# If you add labels that are labeled in the rdf file,
# the training split function will continuously loop.

# 3: Add some sort of parameter visualization.
# 4: Create an executable for this program.
# Note: if you use Pyinstaller, you may want to remove
# images because it requires the executable to be in a
# certain folder to reference images (unless you find 
# a way around it).

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
	# Anything after this line below will execute after the GUI is exited.
	root.mainloop()

	# Want a function/action to be executed continuously?
	# Use act = root.after(time, function)
	# after_cancel(act) to stop it.

# Run the main:
if __name__ == '__main__':
	main()