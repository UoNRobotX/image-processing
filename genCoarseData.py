import sys, re
from PIL import Image, ImageTk
import tkinter

usage = "Usage: python3 " + sys.argv[0] + """
    Reads a list of image filenames is read from stdin.
    Leading and trailing whitespace is ignored.
    Empty names, and names with commas, are ignored.

    Each image is displayed, and the user may mark grid cells.
    Clicking or dragging over a cell toggles whether it is marked as being water.
    Right clicking or dragging over a cell toggles whether it is marked as 'above the horizon'.
        These are cells that will be ignored by the coarse scanner.
        The first right click highlights all non-lower cells.
    Pressing enter causes the next image to be displayed.
    Information about the images and marked cells is written to stdout.
    The output contains lines with the format: imageFile
        'horizon' specifies a cell row, above which cells should be ignored by the coarse scanner.
            0 specifies the top row, 1 specifies the 2nd row, etc.
        Each such line is followed by indented lines for each row, containing 0s, 1s, and 2s.
            A line ' 0112' specifies 4 cells of a row (unmarked, water, water, above horizon)
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
    if len(line) > 0 and line.find(",") == -1:
        filenames.append(line)
if len(filenames) == 0:
    print("No specified filenames", file=sys.stderr)
    sys.exit(1)

#create window
window = tkinter.Tk()
window.title(filenames[filenameIdx])
image = Image.open(filenames[filenameIdx])
canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
canvas.pack()
imageTk = ImageTk.PhotoImage(image)
canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)

#variables
IMG_DOWNSCALE = 2
INPUT_HEIGHT = 32*IMG_DOWNSCALE
INPUT_WIDTH  = 32*IMG_DOWNSCALE
cells = []                #cells[i][j] has this form: [mark, markBoxOrNone, gridBox]
    #'mark' is 0, 1, or 2, indicating unmarked, water, or 'above the horizon'
mouseDownCell = None      #has the form [i,j], specifying the last cell the mouse was in while held down
rightMouseDownCell = None #like 'mouseDownCell', but applies to the right mosue button
firstRightClick = True    #true if no right click has been done yet for the current image

#helper functions
def initCells(): #create new cells, all unmarked, with grid boxes
    global cells
    cells = []
    for i in range(image.size[0]//INPUT_WIDTH):
        cells.append([])
        for j in range(image.size[1]//INPUT_HEIGHT):
            cells[i].append([
                0,
                None,
                canvas.create_rectangle(
                    i*INPUT_WIDTH,
                    j*INPUT_HEIGHT,
                    i*INPUT_WIDTH+INPUT_WIDTH-1,
                    j*INPUT_HEIGHT+INPUT_HEIGHT-1
                )
            ])
def markCell(i, j, mark): #mark cell i-j
    global cells
    if mark == 0:
        canvas.delete(cells[i][j][1])
    elif mark == 1:
        cells[i][j][1] = canvas.create_rectangle(
            i*INPUT_WIDTH,
            j*INPUT_HEIGHT,
            i*INPUT_WIDTH+INPUT_WIDTH-1,
            j*INPUT_HEIGHT+INPUT_HEIGHT-1,
            fill="green",
            stipple='gray50'
        )
    elif mark == 2:
        cells[i][j][1] = canvas.create_rectangle(
            i*INPUT_WIDTH,
            j*INPUT_HEIGHT,
            i*INPUT_WIDTH+INPUT_WIDTH-1,
            j*INPUT_HEIGHT+INPUT_HEIGHT-1,
            fill="orange",
            stipple='gray50'
        )
    else:
        raise Exception("Invalid 'mark' argument")
    cells[i][j][0] = mark

#initialise cells
initCells()

#define mouse/key handlers
def clickCallback(event):
    global mouseDownCell
    i = event.x//INPUT_WIDTH
    j = event.y//INPUT_HEIGHT
    mouseDownCell = [i, j]
    if cells[i][j][0] == 1:
        markCell(i,j,0)
    else:
        markCell(i,j,1)
def rightClickCallback(event):
    global rightMouseDownCell, firstRightClick
    i = event.x//INPUT_WIDTH
    j = event.y//INPUT_HEIGHT
    rightMouseDownCell = [i, j]
    if firstRightClick: #first right click highlights all non-lower cells
        for col in range(len(cells)):
            for row in range(j+1):
                cells[col][row][0] = True
                markCell(col,row,2)
        firstRightClick = False
    else:
        if cells[i][j][0] == 2:
            markCell(i,j,0)
        else:
            markCell(i,j,2)
def moveCallback(event):
    global mouseDownCell
    i = event.x//INPUT_WIDTH
    j = event.y//INPUT_HEIGHT
    if i != mouseDownCell[0] or j != mouseDownCell[1]:
        mouseDownCell = [i, j]
        if cells[i][j][0] == 1:
            markCell(i,j,0)
        else:
            markCell(i,j,1)
def rightMoveCallback(event):
    global rightMouseDownCell
    i = event.x//INPUT_WIDTH
    j = event.y//INPUT_HEIGHT
    if i != rightMouseDownCell[0] or j != rightMouseDownCell[1]:
        rightMouseDownCell = [i, j]
        if cells[i][j][0] == 2:
            markCell(i,j,0)
        else:
            markCell(i,j,2)
def returnCallback(event):
    global filenameIdx, image, imageTk, cells, firstRightClick
    #output info
    print(filenames[filenameIdx])
    for row in range(len(cells[0])):
        print(" ", end="")
        for col in range(len(cells)):
            print(cells[col][row][0], end="")
        print()
    #move to next file, or exit
    filenameIdx += 1
    if filenameIdx < len(filenames):
        window.title(filenames[filenameIdx]) #rename window
        canvas.delete(tkinter.ALL) #remove grid and boxes
        #load new image
        image = Image.open(filenames[filenameIdx])
        canvas.config(width=image.size[0], height=image.size[1])
        canvas.pack()
        imageTk = ImageTk.PhotoImage(image)
        canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)
        initCells()
        firstRightClick = True
    else:
        sys.exit(0)

#bind handlers
canvas.bind("<Button-1>", clickCallback)
canvas.bind("<Button-3>", rightClickCallback)
canvas.bind("<B1-Motion>", moveCallback)
canvas.bind("<B3-Motion>", rightMoveCallback)
window.bind("<Return>", returnCallback)

#start application
tkinter.mainloop()
