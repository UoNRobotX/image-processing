import sys, re
from PIL import Image, ImageTk
import tkinter

usage = "Usage: python3 " + sys.argv[0] + """
    Reads a list of image filenames from stdin.
    Leading and trailing whitespace is ignored. 
    Empty names, and names with commas, are ignored.

    Each image is displayed, and the user may create bounding boxes with the mouse.
    Pressing enter causes the next image to be displayed.
    Information about the images and boxes is written to stdout.
    Output lines have this format: imageFile,topLeftX,topLeftY,bottomRightX,bottomRightY
"""

#check command line arguments
if len(sys.argv) > 1:
    print(usage, file=sys.stderr)
    sys.exit(1)

filenames = []
filenameIdx = 0

#get filenames
for line in sys.stdin:
    line = line.strip()
    if len(line) > 0 and line.find(',') == -1:
        filenames.append(line)
if len(filenames) == 0:
    print('No specified filenames', file=sys.stderr)
    sys.exit(1)

#create window
window = tkinter.Tk()
window.title(filenames[filenameIdx])
image = Image.open(filenames[filenameIdx])
canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
canvas.pack()
imageTk = ImageTk.PhotoImage(image)
canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)

#variables for holding box information
sel = [None, None] #has the form [[x,y], [x,y]], and holds the corners of a box being created
box = None         #holds a the rectangle shown while the mouse is being dragegd
boxes = []         #holds previously created boxes

#define mouse/key handlers
def clickCallback(event):
    sel[0] = [event.x, event.y]
def releaseCallback(event):
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
    global filenameIdx, image, imageTk
    #move to next file, or exit
    filenameIdx += 1
    if filenameIdx < len(filenames):
        window.title(filenames[filenameIdx]) #rename window
        canvas.delete(tkinter.ALL) #remove image and boxes
        #load new image
        image = Image.open(filenames[filenameIdx])
        canvas.config(width=image.size[0], height=image.size[1])
        canvas.pack()
        imageTk = ImageTk.PhotoImage(image)
        canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)
    else:
        sys.exit(0)

#bind handlers
canvas.bind("<Button-1>", clickCallback)
canvas.bind("<ButtonRelease-1>", releaseCallback)
canvas.bind("<B1-Motion>", moveCallback)
window.bind("<Return>", returnCallback)

#start application
tkinter.mainloop()
