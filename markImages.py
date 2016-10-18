import sys, re, os
from PIL import Image, ImageTk
import tkinter

usage = "Usage: python3 " + sys.argv[0] + """ [-f] [-w] [-b] [-d dir1]
    By default, reads a list of image filenames from stdin.
    Leading and trailing whitespace is ignored.
    Empty names, and names with commas, are ignored.

    Each image is displayed, and the user may mark them using the mouse.
    Pressing enter causes the next image to be displayed.
    Information about the images and boxes is written to stdout.

    At least one of -f, -w, or -b should be given.
    If more than one is given, only the last is used.

    Options:
        -b
            The user marks bounding boxes around dark buoys by clicking and dragging.
            The output contains lines holding image filenames.
            Output lines have one of these formats:
                Each such line is followed by indented lines, each specifying a bounding box.
                    A line ' 1,2,3,4' specifies a box with top-left at 1,2 and bottom-right at 3,4.
        -f
            The user marks grid cells that should always be ignored (camera boundaries, roof, etc).
            Clicking or dragging over a cell toggles whether a cell is marked.
            The output contains lines holding image filenames.
                Each such line is followed by indented lines for each row, containing 0s and 1s.
                    A line ' 0111' specifies 4 cells of a row, 3 of which are marked.
        -w
            The user marks grid cells that contain mostly water and nothing else of significance.
            Clicking or dragging over a cell toggles whether a cell is marked.
            The output contains lines for each row of cells, containing 0s and 1s.
                A line ' 0111' specifies 4 cells of a row, 3 of which are marked.
        -d dir1
            Use .jpg files in dir1 as the list of filenames.
"""

#process command line arguments
MODE_BOXES  = 0 #-b
MODE_FILTER = 1 #-f
MODE_WATER  = 2 #-w
mode = None
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
if mode == None:
    print("At least one of -f, -w, or -b should be given")
    sys.exit(1)

#get filenames
filenames = []
filenameIdx = 0
if imageDir == None:
    for line in sys.stdin:
        line = line.strip()
        if len(line) > 0 and line.find(",") == -1:
            filenames.append(line)
else:
    filenames = [
        imageDir + "/" + name for
        name in os.listdir(imageDir) if
        os.path.isfile(imageDir + "/" + name) and re.fullmatch(r".*\.jpg", name)
    ]
    filenames.sort()
if len(filenames) == 0:
    print("No specified filenames", file=sys.stderr)
    sys.exit(1)

#create window
window = tkinter.Tk()
window.title(filenames[filenameIdx])
image = Image.open(filenames[filenameIdx])
canvasWidth = image.size[0]
canvasHeight = image.size[1]
canvas = tkinter.Canvas(window, width=canvasWidth, height=canvasHeight)
canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
imageTk = ImageTk.PhotoImage(image)
canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)

#variables
IMG_DOWNSCALE = 2
INPUT_HEIGHT  = 32*IMG_DOWNSCALE
INPUT_WIDTH   = 32*IMG_DOWNSCALE
sel   = [None, None] #has the form [[x,y], [x,y]], and holds the corners of a box being created
box   = None         #holds a rectangle shown while the mouse is being dragegd
boxes = []           #holds created boxes
cells = []           #cells[i][j] is a canvas box ID or None, indicating if the cell is marked
mouseDownCell = None #has the form [i,j], specifying the last cell the mouse was in while held down

#handler setup functions
def setupMarkBoxHandlers():
    def resizeCallback(event):
        global canvasWidth, canvasHeight, imageTk, canvasImage
        wscale = event.width/canvasWidth
        hscale = event.height/canvasHeight
        canvasWidth = event.width
        canvasHeight = event.height
        canvas.config(width=canvasWidth, height=canvasHeight)
        canvas.scale("all", 0, 0, wscale, hscale)
        canvas.delete(canvasImage)
        imageTk = ImageTk.PhotoImage(
            image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
        )
        canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
        canvas.tag_lower(canvasImage) #move image to back
    def clickCallback(event):
        sel[0] = [event.x, event.y]
    def moveCallback(event):
        global box
        sel[1] = [event.x, event.y]
        canvas.delete(box)
        box = canvas.create_rectangle(
            sel[0][0], sel[0][1], sel[1][0], sel[1][1], outline="red", width=2
        )
    def releaseCallback(event):
        sel[1] = [event.x, event.y]
        canvas.delete(box)
        boxes.append(
            canvas.create_rectangle(
                sel[0][0], sel[0][1], sel[1][0], sel[1][1]
            )
        )
        #convert to non-scaled image coordinates
        wscale = image.size[0]/canvasWidth
        hscale = image.size[1]/canvasHeight
        sel[0][0] = int(sel[0][0] * wscale)
        sel[0][1] = int(sel[0][1] * hscale)
        sel[1][0] = int(sel[1][0] * wscale)
        sel[1][1] = int(sel[1][1] * hscale)
        print(" %d,%d,%d,%d" % (sel[0][0], sel[0][1], sel[1][0], sel[1][1]))
    def returnCallback(event):
        global filenameIdx, image, imageTk, boxes
        #move to next file, or exit
        filenameIdx += 1
        if filenameIdx < len(filenames):
            print(filenames[filenameIdx])
            window.title(filenames[filenameIdx]) #rename window
            canvas.delete(tkinter.ALL) #remove image and boxes
            boxes = []
            #load new image
            image = Image.open(filenames[filenameIdx])
            imageTk = ImageTk.PhotoImage(
                image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
            )
            canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
        else:
            sys.exit(0)
    def escapeCallback(event):
        sys.exit(0)
    print(filenames[filenameIdx])
    canvas.bind("<Configure>", resizeCallback)
    canvas.bind("<Button-1>", clickCallback)
    canvas.bind("<B1-Motion>", moveCallback)
    canvas.bind("<ButtonRelease-1>", releaseCallback)
    window.bind("<Return>", returnCallback)
    window.bind("<Escape>", escapeCallback)
def setupMarkCellHandlers(markFilter):
    #helper functions
    def toggleCell(i, j): #toggle marked-ness of cell i-j
        global cells
        if cells[i][j] != None:
            canvas.delete(cells[i][j])
            cells[i][j] = None
        else:
            wscale = canvasWidth/image.size[0]
            hscale = canvasHeight/image.size[1]
            cells[i][j] = canvas.create_rectangle(
                i * INPUT_WIDTH  * wscale,
                j * INPUT_HEIGHT * hscale,
                (i + 1) * INPUT_WIDTH  * wscale,
                (j + 1) * INPUT_HEIGHT * hscale,
                fill="green",
                stipple="gray50"
            )
    #handlers
    def resizeCallback(event):
        global canvasWidth, canvasHeight, imageTk, canvasImage
        wscale = event.width/canvasWidth
        hscale = event.height/canvasHeight
        canvasWidth = event.width
        canvasHeight = event.height
        canvas.config(width=canvasWidth, height=canvasHeight)
        canvas.scale("all", 0, 0, wscale, hscale)
        canvas.delete(canvasImage)
        imageTk = ImageTk.PhotoImage(
            image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
        )
        canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
        canvas.tag_lower(canvasImage) #move image to back
    def clickCallback(event):
        global mouseDownCell
        wscale = image.size[0]/canvasWidth
        hscale = image.size[1]/canvasHeight
        i = int(event.x * wscale) // INPUT_WIDTH
        j = int(event.y * hscale) // INPUT_HEIGHT
        mouseDownCell = [i, j]
        toggleCell(i,j)
    def moveCallback(event):
        global mouseDownCell
        #do nothing if mouse is outside window
        if event.x < 0 or event.x > canvasWidth-1 or event.y < 0 or event.y > canvasHeight-1:
            return
        wscale = image.size[0]/canvasWidth
        hscale = image.size[1]/canvasHeight
        i = int(event.x * wscale) // INPUT_WIDTH
        j = int(event.y * hscale) // INPUT_HEIGHT
        if i != mouseDownCell[0] or j != mouseDownCell[1]:
            mouseDownCell = [i, j]
            toggleCell(i,j)
    def markFilterReturnCallback(event):
        global filenameIdx, image, imageTk, canvasImage, boxes, cells
        #move to next file, or exit
        filenameIdx += 1
        if filenameIdx < len(filenames):
            window.title(filenames[filenameIdx]) #rename window
            #load new image
            canvas.delete(canvasImage)
            image = Image.open(filenames[filenameIdx])
            imageTk = ImageTk.PhotoImage(
                image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
            )
            canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
            canvas.tag_lower(canvasImage) #move image to back
        else:
            #output filter info
            for row in range(len(cells[0])):
                for col in range(len(cells)):
                    print("0" if cells[col][row] == None else "1", end="")
                print()
            sys.exit(0)
    def markWaterReturnCallback(event):
        global filenameIdx, image, imageTk, canvasImage, boxes, cells
        #output info
        print(filenames[filenameIdx])
        for row in range(len(cells[0])):
            print(" ", end="")
            for col in range(len(cells)):
                print("0" if cells[col][row] == None else "1", end="")
            print()
        #move to next file, or exit
        filenameIdx += 1
        if filenameIdx < len(filenames):
            window.title(filenames[filenameIdx]) #rename window
            #remove colored boxes
            for i in range(len(cells)):
                for j in range(len(cells[i])):
                    canvas.delete(cells[i][j])
                    cells[i][j] = None
            #load new image
            canvas.delete(canvasImage)
            image = Image.open(filenames[filenameIdx])
            imageTk = ImageTk.PhotoImage(
                image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
            )
            canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
            canvas.tag_lower(canvasImage) #move image to back
        else:
            sys.exit(0)
    def markFilterEscapeCallback(event):
        #output filter info
        for row in range(len(cells[0])):
            for col in range(len(cells)):
                print("0" if cells[col][row] == None else "1", end="")
            print()
        sys.exit(0)
    def markWaterEscapeCallback(event):
        #output info
        print(filenames[filenameIdx])
        for row in range(len(cells[0])):
            print(" ", end="")
            for col in range(len(cells)):
                print("0" if cells[col][row] == None else "1", end="")
            print()
        sys.exit(0)
    #set handlers
    canvas.bind("<Configure>", resizeCallback)
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
else:
    #initialise cells
    cells = [
        [None for row in range(image.size[1]//INPUT_HEIGHT)]
        for col in range(image.size[0]//INPUT_WIDTH)
    ]
    #create cell outlines
    for i in range(len(cells)):
        for j in range(len(cells[0])):
            boxes.append(
                canvas.create_rectangle(
                    i*INPUT_WIDTH,
                    j*INPUT_HEIGHT,
                    (i+1)*INPUT_WIDTH,
                    (j+1)*INPUT_HEIGHT
                )
            )
    #setup handlers
    if mode == MODE_FILTER:
        setupMarkCellHandlers(True)
    elif mode == MODE_WATER:
        setupMarkCellHandlers(False)

#start application
tkinter.mainloop()
