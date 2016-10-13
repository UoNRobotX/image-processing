import tkinter
from PIL import Image, ImageTk
import sys, re

usage = "Usage: python3 " + sys.argv[0] + """
    Reads a list of image filenames is read from stdin.
    Leading and trailing whitespace is ignored. 
    Empty names, and names with commas, are ignored.

    Each image is displayed, and the user may create bounding boxes with the mouse.
    Pressing enter causes the next image to be displayed.
    Information about the images and boxes is written to stdout.
"""
if len(sys.argv) > 1:
	print(usage, file=sys.stderr)
	sys.exit(1)

filenames = []
for line in sys.stdin:
	line = line.strip()
	if len(line) == 0 or line.find(',') != -1:
		continue
	filenames.append(line.strip())

window = tkinter.Tk()

filenameIdx = 0
image = Image.open(filenames[filenameIdx])
window.title(filenames[filenameIdx])
canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
canvas.pack()
image_tk = ImageTk.PhotoImage(image)
canvasImage = canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)

sel = [None, None]
box = None
boxes = []

def clickCallback(event):
	#print("clicked at: ", event.x, event.y)
	sel[0] = [event.x, event.y]
def releaseCallback(event):
	#print("released at: ", event.x, event.y)
	sel[1] = [event.x, event.y]
	canvas.delete(box)
	boxes.append(
		canvas.create_rectangle(
			sel[0][0], sel[0][1], sel[1][0], sel[1][1]
		)
	)
	print('%s,%d,%d,%d,%d' % (filenames[filenameIdx], sel[0][0], sel[0][1], sel[1][0], sel[1][1]))
def moveCallback(event):
	global box
	sel[1] = [event.x, event.y]
	canvas.delete(box)
	box = canvas.create_rectangle(
		sel[0][0], sel[0][1], sel[1][0], sel[1][1], outline='red', width=2
	)
def returnCallback(event):
	global filenameIdx, image, image_tk, canvasImage
	filenameIdx += 1
	window.title(filenames[filenameIdx])
	if filenameIdx < len(filenames):
		#remove boxes
		canvas.delete(box)
		for i in range(len(boxes)):
			canvas.delete(boxes[i])
		#load new image
		canvas.delete(canvasImage)
		image = Image.open(filenames[filenameIdx])
		canvas.config(width=image.size[0], height=image.size[1])
		canvas.pack()
		image_tk = ImageTk.PhotoImage(image)
		canvasImage = canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)
	else:
		sys.exit(0)

canvas.bind("<Button-1>", clickCallback)
canvas.bind("<ButtonRelease-1>", releaseCallback)
canvas.bind("<B1-Motion>", moveCallback)
window.bind("<Return>", returnCallback)
tkinter.mainloop()
