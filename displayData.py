import tkinter
from PIL import Image, ImageTk, ImageDraw
import sys, re

usage = "Usage: python3 " + sys.argv[0] + """ -s file1
    Reads file1, which specifies images and boxes, and displays them.
    file1 should have lines of the form: imageFile,topLeftX,topLeftY,bottomRightX,bottomRightY.
    Pressing enter causes the next image to be displayed.

    Options:
        -s
            Save images to files. A file 'images/img1.jpg' is saved as 'images/img1_boxed.jpg'.
"""
dataFile = None
saveFiles = False
for i in range(1, len(sys.argv)):
	arg = sys.argv[i]
	if arg == '-s':
		saveFiles = True
	else:
		dataFile = arg
if dataFile == None:
	print(usage, file=sys.stderr)

lines = None
with open(dataFile) as file:
	lines = file.readlines()

records = []
for i in range(len(lines)):
	record = lines[i].strip().split(',')
	if len(record) < 5:
		print('Warning: a line has too few fields')
		continue
	records.append([record[0], int(record[1]), int(record[2]), int(record[3]), int(record[4])])

if len(records) == 0: 
	sys.exit(0)

window = tkinter.Tk()

filename = records[0][0]
recordIdx = 0
window.title(filename)
image = Image.open(filename)
draw = ImageDraw.Draw(image)
canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
canvas.pack()
image_tk = ImageTk.PhotoImage(image)
canvasImage = canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)

boxes = []

while (recordIdx < len(records)):
	record = records[recordIdx]
	if (record[0] == filename):
		boxes.append(
			canvas.create_rectangle(
				record[1], record[2], record[3], record[4], outline='red', width=2
			)
		)
		draw.rectangle([record[1], record[2], record[3], record[4]], outline=(255,0,0))
	else:
		break
	recordIdx += 1

def returnCallback(event):
	global filename, recordIdx, image, draw, image_tk, canvasImage, boxes
	#save file
	if saveFiles:
		image.save(re.sub(r'(\.[^.]*)?$', r'_boxed\1', filename))
	if recordIdx < len(records):
		#remove boxes
		for i in range(len(boxes)):
			canvas.delete(boxes[i])
		boxes = []
		#load new image
		canvas.delete(canvasImage)
		filename = records[recordIdx][0]
		window.title(filename)
		image = Image.open(filename)
		draw = ImageDraw.Draw(image)
		canvas.config(width=image.size[0], height=image.size[1])
		canvas.pack()
		image_tk = ImageTk.PhotoImage(image)
		canvasImage = canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)
		#load boxes
		while (recordIdx < len(records)):
			record = records[recordIdx]
			if (record[0] == filename):
				boxes.append(
					canvas.create_rectangle(
						record[1], record[2], record[3], record[4], outline='red', width=2
					)
				)
				draw.rectangle([record[1], record[2], record[3], record[4]], outline=(255,0,0))
			else:
				break
			recordIdx += 1
	else:
		sys.exit(0)

window.bind("<Return>", returnCallback)
tkinter.mainloop()
