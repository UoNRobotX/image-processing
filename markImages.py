import sys, re, os
from PIL import Image, ImageTk
import tkinter

usage = "Usage: python3 " + sys.argv[0] + """[-f] [-w] [-b] [-d dir1]
    By default, reads a list of image filenames from stdin.
    Leading and trailing whitespace is ignored.
    Empty names, and names with commas, are ignored.

    Each image is displayed, and the user may mark them using the mouse.
    Pressing enter causes the next image to be displayed.
    Information about the images and boxes is written to stdout.

    By default, -b is implied.
    If more than one of -f, -w, -b is given, only the last is used.

    Options:
        -b
            The user marks bounding boxes around dark buoys by clicking and dragging.
            Output lines have this format: imageFile,topLeftX,topLeftY,bottomRightX,bottomRightY
        -f
            The user marks grid cells that should always be ignored (camera boundaries, roof, etc).
            Clicking or dragging over a cell toggles whether a cell is marked.
            The output contains lines holding image filenames.
                Each such line is followed by indented lines for each row, containing 0s and 1s.
                    A line ' 0111' specifies 4 cells of a row, 3 of which are marked.
        -w
            The user marks grid cells that contain mostly water and nothing else of significance.
            Clicking or dragging over a cell toggles whether a cell is marked.
            Output is similar to that produced by -f.
        -d dir1
            Use .jpg files in dir1 as the list of filenames.
"""

#process command line arguments
MODE_BOXES  = 0 #-b
MODE_FILTER = 1 #-f
MODE_WATER  = 2 #-w
mode = MODE_BOXES
imageDir = None
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "-b":
        mode = MODE_BOXES
    elif arg == "-f":
        mode = MODE_FILTER
    elif arg == "-w":
        mode = MODE_WATER
    elif arg == "-d":
        i += 1
        if i < len(sys.argv):
            imageDir = sys.argv[i]
        else:
            print("No argument for -d", file=sys.stderr)
            sys.exit(1)
    else:
        print(usage, file=sys.stderr)
        sys.exit(1)
    i += 1

#get filenames
filenames = []
filenameIdx = 0
if imageDir == None:
    for line in sys.stdin:
        line = line.strip()
        if len(line) > 0 and line.find(',') == -1:
            filenames.append(line)
else:
    filenames = [
        imageDir + "/" + name for
        name in os.listdir(imageDir) if
        os.path.isfile(imageDir + "/" + name) and re.fullmatch(r".*\.jpg", name)
    ]
    filenames.sort()
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

#variables
IMG_DOWNSCALE = 2
INPUT_HEIGHT  = 32*IMG_DOWNSCALE
INPUT_WIDTH   = 32*IMG_DOWNSCALE
sel   = [None, None] #has the form [[x,y], [x,y]], and holds the corners of a box being created
box   = None         #holds a rectangle shown while the mouse is being dragegd
boxes = []           #holds created boxes
cells = []           #cells[i][j] has this form: [mark, markBoxOrNone], where 'mark' is 0 or 1
mouseDownCell = None #has the form [i,j], specifying the last cell the mouse was in while held down

#handler setup functions
def setupMarkBoxHandlers():
    def clickCallback(event):
        sel[0] = [event.x, event.y]
    def moveCallback(event):
        global box
        sel[1] = [event.x, event.y]
        canvas.delete(box)
        box = canvas.create_rectangle(
            sel[0][0], sel[0][1], sel[1][0], sel[1][1], outline='red', width=2
        )
    def releaseCallback(event):
        sel[1] = [event.x, event.y]
        canvas.delete(box)
        boxes.append(
            canvas.create_rectangle(
                sel[0][0], sel[0][1], sel[1][0], sel[1][1]
            )
        )
        print('%s,%d,%d,%d,%d' % (filenames[filenameIdx], sel[0][0], sel[0][1], sel[1][0], sel[1][1]))
    def returnCallback(event):
        global filenameIdx, image, imageTk, boxes
        #move to next file, or exit
        filenameIdx += 1
        if filenameIdx < len(filenames):
            window.title(filenames[filenameIdx]) #rename window
            canvas.delete(tkinter.ALL) #remove image and boxes
            boxes = []
            #load new image
            image = Image.open(filenames[filenameIdx])
            canvas.config(width=image.size[0], height=image.size[1])
            canvas.pack()
            imageTk = ImageTk.PhotoImage(image)
            canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)
        else:
            sys.exit(0)
    def escapeCallback(event):
        sys.exit(0)
    canvas.bind("<Button-1>", clickCallback)
    canvas.bind("<B1-Motion>", moveCallback)
    canvas.bind("<ButtonRelease-1>", releaseCallback)
    window.bind("<Return>", returnCallback)
    window.bind("<Escape>", escapeCallback)
def setupMarkCellHandlers(markFilter):
    #helper functions
    def initCells(): #create new cells, unmarked, with grid boxes
        global cells
        for i in range(image.size[0]//INPUT_WIDTH):
            cells.append([])
            for j in range(image.size[1]//INPUT_HEIGHT):
                cells[i].append([0, None])
                boxes.append(
                    canvas.create_rectangle(
                        i*INPUT_WIDTH,
                        j*INPUT_HEIGHT,
                        i*INPUT_WIDTH+INPUT_WIDTH-1,
                        j*INPUT_HEIGHT+INPUT_HEIGHT-1
                    )
                )
    def toggleCell(i, j): #toggle marked-ness of cell i-j
        global cells
        if cells[i][j][0] == 1:
            canvas.delete(cells[i][j][1])
            cells[i][j] = [0, None]
        else:
            cells[i][j] = [
                1,
                canvas.create_rectangle(
                    i*INPUT_WIDTH,
                    j*INPUT_HEIGHT,
                    i*INPUT_WIDTH+INPUT_WIDTH-1,
                    j*INPUT_HEIGHT+INPUT_HEIGHT-1,
                    fill="green",
                    stipple="gray50"
                )
            ]
    #handlers
    def clickCallback(event):
        global mouseDownCell
        i = event.x//INPUT_WIDTH
        j = event.y//INPUT_HEIGHT
        mouseDownCell = [i, j]
        toggleCell(i,j)
    def moveCallback(event):
        global mouseDownCell
        i = event.x//INPUT_WIDTH
        j = event.y//INPUT_HEIGHT
        if i != mouseDownCell[0] or j != mouseDownCell[1]:
            mouseDownCell = [i, j]
            toggleCell(i,j)
    def markFilterReturnCallback(event):
        global filenameIdx, image, imageTk, boxes, cells
        #move to next file, or exit
        filenameIdx += 1
        if filenameIdx < len(filenames):
            window.title(filenames[filenameIdx]) #rename window
            canvas.delete(tkinter.ALL) #remove grid and boxes
            boxes = []
            #load new image
            image = Image.open(filenames[filenameIdx])
            canvas.config(width=image.size[0], height=image.size[1])
            canvas.pack()
            imageTk = ImageTk.PhotoImage(image)
            canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)
            #redraw grid and boxes
            for i in range(image.size[0]//INPUT_WIDTH):
                for j in range(image.size[1]//INPUT_HEIGHT):
                    boxes.append(
                        canvas.create_rectangle(
                            i*INPUT_WIDTH,
                            j*INPUT_HEIGHT,
                            i*INPUT_WIDTH+INPUT_WIDTH-1,
                            j*INPUT_HEIGHT+INPUT_HEIGHT-1
                        )
                    )
                    if cells[i][j][0]:
                        cells[i][j][1] = canvas.create_rectangle(
                            i*INPUT_WIDTH,
                            j*INPUT_HEIGHT,
                            i*INPUT_WIDTH+INPUT_WIDTH-1,
                            j*INPUT_HEIGHT+INPUT_HEIGHT-1,
                            fill="orange",
                            stipple="gray50"
                        )
        else:
            #output filter info
            for row in range(len(cells[0])):
                for col in range(len(cells)):
                    print(cells[col][row][0], end="")
                print()
            sys.exit(0)
    def markWaterReturnCallback(event):
        global filenameIdx, image, imageTk, boxes, cells
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
            cells = []
            boxes = []
            #load new image
            image = Image.open(filenames[filenameIdx])
            canvas.config(width=image.size[0], height=image.size[1])
            canvas.pack()
            imageTk = ImageTk.PhotoImage(image)
            canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)
            initCells()
        else:
            sys.exit(0)
    def markFilterEscapeCallback(event):
        #output filter info
        for row in range(len(cells[0])):
            for col in range(len(cells)):
                print(cells[col][row][0], end="")
            print()
        sys.exit(0)
    def markWaterEscapeCallback(event):
        #output info
        print(filenames[filenameIdx])
        for row in range(len(cells[0])):
            print(" ", end="")
            for col in range(len(cells)):
                print(cells[col][row][0], end="")
            print()
        sys.exit(0)
    initCells()
    canvas.bind("<Button-1>", clickCallback)
    canvas.bind("<B1-Motion>", moveCallback)
    if markFilter:
        window.bind("<Return>", markFilterReturnCallback)
        window.bind("<Escape>", markFilterEscapeCallback)
    else:
        window.bind("<Return>", markWaterReturnCallback)
        window.bind("<Escape>", markWaterEscapeCallback)

#setup
if mode == MODE_BOXES:
    setupMarkBoxHandlers()
elif mode == MODE_FILTER:
    setupMarkCellHandlers(True)
elif mode == MODE_WATER:
    setupMarkCellHandlers(False)

#start application
tkinter.mainloop()
