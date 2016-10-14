import sys, re
from PIL import Image, ImageTk, ImageDraw
import tkinter

usage = "Usage: python3 " + sys.argv[0] + """ [-s] file1
    Reads file1, which specifies images and boxes, and displays them.
    file1 should have the same format as output by genData.py.
    Pressing enter causes the next image to be displayed.

    Options:
        -s
            Save images to files. A file 'images/img1.jpg' is saved as 'images/img1_boxed.jpg'.
"""

#process command line arguments
dataFile = None   #the file to read info from
saveFiles = False #True if image files are to be saved
for i in range(1, len(sys.argv)):
	arg = sys.argv[i]
	if arg == '-s':
		saveFiles = True
	else:
		dataFile = arg
if dataFile == None:
	print(usage, file=sys.stderr)
	sys.exit(1)

#read data file
records = []
recordIdx = 0
with open(dataFile) as file:
	for line in file:
		record = line.strip().split(',')
		if len(record) < 5:
			print('Warning: a line has too few fields', file=sys.stderr)
			continue
		records.append([record[0], int(record[1]), int(record[2]), int(record[3]), int(record[4])])
if len(records) == 0: 
	sys.exit(0)
filename = records[0][0]

#create window
window = tkinter.Tk()
window.title(filename)
image = Image.open(filename)
draw = ImageDraw.Draw(image)
canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
canvas.pack()
imageTk = ImageTk.PhotoImage(image)
canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)

#helper functions
def drawBoxes():
	global recordIdx
	while (recordIdx < len(records)):
		record = records[recordIdx]
		if record[0] == filename:
			canvas.create_rectangle(
				record[1], record[2], record[3], record[4], outline='red', width=2
			)
			draw.rectangle([record[1], record[2], record[3], record[4]], outline=(255,0,0))
		else:
			break
		recordIdx += 1

#draw boxes for current image file
drawBoxes()

#define enter key handler
def returnCallback(event):
	global filename, recordIdx, image, draw, imageTk
	#save file if requested
	if saveFiles:
		image.save(re.sub(r'(\.[^.]*)?$', r'_boxed\1', filename))
	#move to next image, or exit
	if recordIdx < len(records):
		#remove image and boxes
		canvas.delete(tkinter.ALL)
		#load new image
		filename = records[recordIdx][0]
		window.title(filename)
		image = Image.open(filename)
		draw = ImageDraw.Draw(image)
		canvas.config(width=image.size[0], height=image.size[1])
		canvas.pack()
		imageTk = ImageTk.PhotoImage(image)
		canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)
		drawBoxes()
	else:
		sys.exit(0)

#bind handlers
window.bind("<Return>", returnCallback)

#start application
tkinter.mainloop()
