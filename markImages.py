import sys, re, os
from PIL import Image, ImageTk, ImageDraw
import tkinter

usage = "Usage: python3 " + sys.argv[0] + """ [-f] [-w] [-b] [-d d1] [-o f1] [-l f1] [-g f1] [-s d1]
    Obtains a list of image filenames, specified by -d, or from stdin.
    If obtained from stdin, each line should contain a filename.
        Leading and trailing whitespace, empty names, and names with commas, are ignored.

    Each image is displayed, and the user may mark them using the mouse.
    Pressing right/left causes the next/previous image to be displayed.
    Information about the images and boxes is written to stdout.

    At least one of -f, -w, or -b should be given.
    If more than one is given, only the last is used.

    Options:
        -d d1
            Use .jpg files in directory d1 as the list of filenames.
        -o f1
            Write output to file f1 instead of to stdout.
        -f
            Mark grid cells that should always be ignored (camera boundaries, roof, etc).
            Clicking or dragging over a cell toggles whether it is marked.
            The output contains lines for each row of cells, containing 0s and 1s.
                A line ' 0111' specifies 4 cells of a row, 3 of which are marked.
        -w
            Mark grid cells that contain only water.
            Clicking or dragging over a cell toggles whether it is marked.
            The output contains sections, each describing cells to ignore for an image.
                Each section starts with a line containing the image filename.
                    The line is followed by indented lines for each row, containing 0s and 1s.
                    A line ' 0111' specifies 4 cells of a row, 3 of which are marked.
        -b
            The user marks bounding boxes by clicking and dragging.
            Boxes can be deleted by right-clicking.
            The output contains lines holding image filenames.
            The output contains sections, each describing boxes for an image.
                Each section starts with a line containing the image filename.
                    The line is followed by indented lines, each specifying a bounding box.
                    A line ' 1,2,3,4' specifies a box with top-left at 1,2 and bottom-right at 3,4.
        -l f1
            Read markings from file f1, whose format should correspond to that expected by -f/-w/-b.
        -g f1
            Skip to file f1 in the list.
        -s d1
            Save the images, with markings, to directory d1.
"""

#process command line arguments
MODE_FILTER = 0 # -f
MODE_WATER  = 1 # -w
MODE_BOXES  = 2 # -b
mode = None
inputDir = None
dataFile = None
outputFile = None
skipFile = None
outputDir = None
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "-f":
        mode = MODE_FILTER
    elif arg == "-w":
        mode = MODE_WATER
    elif arg == "-b":
        mode = MODE_BOXES
    elif arg == "-d":
        i += 1
        if i < len(sys.argv):
            inputDir = sys.argv[i]
        else:
            print("No argument for -d", file=sys.stderr)
            sys.exit(1)
    elif arg == "-o":
        i += 1
        if i < len(sys.argv):
            outputFile = sys.argv[i]
        else:
            print("No argument for -o", file=sys.stderr)
            sys.exit(1)
    elif arg == "-l":
        i += 1
        if i < len(sys.argv):
            dataFile = sys.argv[i]
        else:
            print("No argument for -l", file=sys.stderr)
            sys.exit(1)
    elif arg == "-g":
        i += 1
        if i < len(sys.argv):
            skipFile = sys.argv[i]
        else:
            print("No argument for -g", file=sys.stderr)
            sys.exit(1)
    elif arg == "-s":
        i += 1
        if i < len(sys.argv):
            outputDir = sys.argv[i]
            if not os.path.isdir(outputDir):
                print("Non-directory argument for -s", file=sys.stderr)
                sys.exit(1)
        else:
            print("No argument for -s", file=sys.stderr)
            sys.exit(1)
    else:
        print(usage, file=sys.stderr)
        sys.exit(1)
    i += 1
if mode == None:
    print("At least one of -f, -w, or -b should be given")
    sys.exit(1)

#get input filenames
filenames = dict() #maps each filename to None, or a list containing mark information
filenameIdx = 0
if inputDir == None:
    for line in sys.stdin:
        line = line.strip()
        if len(line) > 0 and line.find(",") == -1:
            filenames[line] = None
else:
    filenames = {
        (inputDir + "/" + name) : None for
        name in os.listdir(inputDir) if
        os.path.isfile(inputDir + "/" + name) and re.fullmatch(r".*\.jpg", name)
    }
if len(filenames) == 0:
    print("No input files", file=sys.stderr)
    sys.exit(1)

#load markings if specified
cellFilter = None #contains loaded filter information
if dataFile != None:
    with open(dataFile) as file:
        if mode == MODE_FILTER:
            cellFilter = []
            for line in file:
                cellFilter.append([int(c) for c in line.strip()])
        elif mode == MODE_WATER:
            filename = None
            for line in file:
                if line[0] != " ":
                    filename = line.strip()
                    filenames[filename] = []
                else:
                    filenames[filename].append([int(c) for c in line.strip()])
        elif mode == MODE_BOXES:
            filename = None
            for line in file:
                if line[0] != " ":
                    filename = line.strip()
                    filenames[filename] = []
                elif filename != None:
                    filenames[filename].append([int(c) for c in line.strip().split(",")])
                else:
                    print("Invalid boxes file", file=sys.stderr)
                    sys.exit(1)

#for iterating through filenames
filenameIdx = 0
filenamesSorted = [name for name in filenames]
filenamesSorted.sort()
if skipFile != None:
    found = False
    for i in range(len(filenamesSorted)):
        if os.path.basename(filenamesSorted[i]) == skipFile:
            filenameIdx = i
            found = True
    if not found:
        print("Skip file not found", file=sys.stderr)
        sys.exit(1)

#variables
DOWNSCALE     = 2
INPUT_HEIGHT  = 32
INPUT_WIDTH   = 32
sel           = [None, None] #[[x,y], [x,y]], holds the corners of a box being created
box           = None         #holds a rectangle shown while the mouse is being dragged
boxCoords     = []           #[[x,y,x,y], ...], describes created bounding boxes or grid cells
boxIDs        = []           #holds IDs of boxes in 'boxCoords'
cells         = []           #cells[i][j] is a canvas box ID or None, indicating if the cell is marked
mouseDownCell = [0,0]        #[i,j], specifies the last cell the mouse was in while held down

#create window
window = tkinter.Tk()
window.title(filenamesSorted[filenameIdx])
image = Image.open(filenamesSorted[filenameIdx])
image = image.resize((image.size[0]//DOWNSCALE, image.size[1]//DOWNSCALE), resample=Image.LANCZOS)
canvasWidth = image.size[0]
canvasHeight = image.size[1]
canvas = tkinter.Canvas(window, width=canvasWidth, height=canvasHeight)
canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
imageTk = ImageTk.PhotoImage(image)
canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)

#handler setup functions
def setupMarkCell(markFilter):
    #helper functions
    def toggleCell(i, j): #toggle marked-ness of cell i-j
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
        mouseDownCell[:] = [i, j]
        toggleCell(i,j)
    def moveCallback(event):
        #do nothing if mouse is outside window
        if event.x < 0 or event.x > canvasWidth-1 or event.y < 0 or event.y > canvasHeight-1:
            return
        wscale = image.size[0]/canvasWidth
        hscale = image.size[1]/canvasHeight
        i = int(event.x * wscale) // INPUT_WIDTH
        j = int(event.y * hscale) // INPUT_HEIGHT
        if i != mouseDownCell[0] or j != mouseDownCell[1]:
            mouseDownCell[:] = [i, j]
            toggleCell(i,j)
    def markFilterNextCallback(event, forward=True):
        global filenameIdx, image, imageTk, canvasImage, cells
        #move to next file, or exit
        filenameIdx = filenameIdx+1 if forward else max(filenameIdx-1, 0)
        if filenameIdx < len(filenamesSorted):
            window.title(filenamesSorted[filenameIdx]) #rename window
            #load new image
            canvas.delete(canvasImage)
            image = Image.open(filenamesSorted[filenameIdx])
            image = image.resize((image.size[0]//DOWNSCALE, image.size[1]//DOWNSCALE), resample=Image.LANCZOS)
            imageTk = ImageTk.PhotoImage(
                image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
            )
            canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
            canvas.tag_lower(canvasImage) #move image to back
        else:
            markWaterEscapeCallback(None)
    def markFilterPrevCallback(event):
        markFilterNextCallback(None, forward=False)
    def markWaterNextCallback(event, forward=True):
        global filenameIdx, image, imageTk, canvasImage, cells
        #store info
        info = [
            [0 if cells[col][row] == None else 1 for col in range(len(cells))]
            for row in range(len(cells[0]))
        ]
        filenames[filenamesSorted[filenameIdx]] = info
        #move to next file, or exit
        filenameIdx = filenameIdx+1 if forward else max(filenameIdx-1, 0)
        if filenameIdx < len(filenamesSorted):
            filename = filenamesSorted[filenameIdx]
            window.title(filename) #rename window
            #remove colored boxes
            for i in range(len(cells)):
                for j in range(len(cells[i])):
                    canvas.delete(cells[i][j])
                    cells[i][j] = None
            #load colored boxes if present
            if filenames[filename] != None:
                info = filenames[filename]
                for row in range(len(info)):
                    for col in range(len(info[0])):
                        if info[row][col] == 1:
                            toggleCell(col, row)
            #load new image
            canvas.delete(canvasImage)
            image = Image.open(filename)
            image = image.resize((image.size[0]//DOWNSCALE, image.size[1]//DOWNSCALE), resample=Image.LANCZOS)
            imageTk = ImageTk.PhotoImage(
                image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
            )
            canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
            canvas.tag_lower(canvasImage) #move image to back
        else:
            markWaterEscapeCallback(None)
    def markWaterPrevCallback(event):
        markWaterNextCallback(None, forward=False)
    def markFilterEscapeCallback(event=None):
        #output filter info
        f = sys.stdout if outputFile == None else open(outputFile, 'w')
        for row in range(len(cells[0])):
            for col in range(len(cells)):
                print("0" if cells[col][row] == None else "1", end="", file=f)
            print(file=f)
        if not f is sys.stdout:
            f.close()
        #save images if requested
        if outputDir != None:
            for filename in filenamesSorted:
                image = Image.open(filename)
                draw = ImageDraw.Draw(image, "RGBA")
                for i in range(image.size[0]//DOWNSCALE//INPUT_WIDTH):
                    for j in range(image.size[1]//DOWNSCALE//INPUT_HEIGHT):
                        topLeftX = i * INPUT_WIDTH * DOWNSCALE
                        topLeftY = j * INPUT_HEIGHT * DOWNSCALE
                        bottomRightX = (i+1) * INPUT_WIDTH  * DOWNSCALE - 1
                        bottomRightY = (j+1) * INPUT_HEIGHT * DOWNSCALE - 1
                        #draw grid box
                        draw.rectangle(
                            [topLeftX, topLeftY, bottomRightX, bottomRightY],
                            outline=(0,0,0)
                        )
                        #draw marking
                        if cells[i][j] != None:
                            draw.rectangle(
                                [topLeftX, topLeftY, bottomRightX, bottomRightY],
                                fill=(0,128,0,128)
                            )
                image.save(outputDir + "/" + os.path.basename(filename))
        sys.exit(0)
    def markWaterEscapeCallback(event=None):
        #store info
        if filenameIdx < len(filenamesSorted):
            info = [
                [0 if cells[col][row] == None else 1 for col in range(len(cells))]
                for row in range(len(cells[0]))
            ]
            filenames[filenamesSorted[filenameIdx]] = info
        #output info
        f = sys.stdout if outputFile == None else open(outputFile, 'w')
        for filename in filenamesSorted:
            if filenames[filename] != None:
                print(filename, file=f)
                info = filenames[filename] 
                for row in range(len(info)):
                    print(" ", end="", file=f)
                    for col in range(len(info[row])):
                        print("0" if info[row][col] == 0 else "1", end="", file=f)
                    print(file=f)
        if not f is sys.stdout:
            f.close()
        #save images if requested
        if outputDir != None:
            for filename in filenamesSorted:
                info = filenames[filename]
                if info != None:
                    image = Image.open(filename)
                    draw = ImageDraw.Draw(image, "RGBA")
                    for i in range(image.size[0]//DOWNSCALE//INPUT_WIDTH):
                        for j in range(image.size[1]//DOWNSCALE//INPUT_HEIGHT):
                            topLeftX = i * INPUT_WIDTH * DOWNSCALE
                            topLeftY = j * INPUT_HEIGHT * DOWNSCALE
                            bottomRightX = (i+1) * INPUT_WIDTH  * DOWNSCALE - 1
                            bottomRightY = (j+1) * INPUT_HEIGHT * DOWNSCALE - 1
                            #draw grid box
                            draw.rectangle(
                                [topLeftX, topLeftY, bottomRightX, bottomRightY],
                                outline=(0,0,0)
                            )
                            #draw marking
                            if info[j][i] == 1:
                                draw.rectangle(
                                    [topLeftX, topLeftY, bottomRightX, bottomRightY],
                                    fill=(0,128,0,128)
                                )
                    image.save(outputDir + "/" + os.path.basename(filename))
        sys.exit(0)
    #set handlers
    canvas.bind("<Configure>", resizeCallback)
    canvas.bind("<Button-1>", clickCallback)
    canvas.bind("<B1-Motion>", moveCallback)
    if markFilter:
        #load cell filter if provided
        if cellFilter != None:
            for row in range(len(cellFilter)):
                for col in range(len(cellFilter[0])):
                    if cellFilter[row][col] == 1:
                        toggleCell(col, row)
        #set more handlers
        window.bind("<Return>", markFilterNextCallback)
        window.bind("<Right>", markFilterNextCallback)
        window.bind("<Left>", markFilterPrevCallback)
        window.bind("<Escape>", markFilterEscapeCallback)
        window.protocol("WM_DELETE_WINDOW", markFilterEscapeCallback)
    else:
        #load water cells if provided
        filename = filenamesSorted[filenameIdx]
        if filenames[filename] != None:
            info = filenames[filename]
            for row in range(len(info)):
                for col in range(len(info[0])):
                    if info[row][col] == 1:
                        toggleCell(col, row)
        #set more handlers
        window.bind("<Return>", markWaterNextCallback)
        window.bind("<Right>", markWaterNextCallback)
        window.bind("<Left>", markWaterPrevCallback)
        window.bind("<Escape>", markWaterEscapeCallback)
        window.protocol("WM_DELETE_WINDOW", markWaterEscapeCallback)
def setupMarkBox():
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
        sel[0] = [event.x, event.y]
    def moveCallback(event):
        global box
        sel[1] = [event.x, event.y]
        canvas.delete(box)
        box = canvas.create_rectangle(
            sel[0][0], sel[0][1], sel[1][0], sel[1][1], outline="red", width=2
        )
    def releaseCallback(event):
        global box
        sel[1] = [event.x, event.y]
        canvas.delete(box)
        #ignore if box is too small
        MIN_SIZE = 5
        if abs(sel[0][0] - sel[1][0]) < MIN_SIZE or abs(sel[0][1] - sel[1][1]) < MIN_SIZE:
            return
        #make 'sel' contain top-left and bottom-right
        if sel[0][0] > sel[1][0]:
            temp = sel[0][0]
            sel[0][0] = sel[1][0]
            sel[1][0] = temp
        if sel[0][1] > sel[1][1]:
            temp = sel[0][1]
            sel[0][1] = sel[1][1]
            sel[1][1] = temp
        #add box
        boxIDs.append(
            canvas.create_rectangle(sel[0][0], sel[0][1], sel[1][0], sel[1][1])
        )
        #convert to non-scaled image coordinates
        wscale = (image.size[0] * DOWNSCALE)/canvasWidth
        hscale = (image.size[1] * DOWNSCALE)/canvasHeight
        sel[0][0] = int(sel[0][0] * wscale)
        sel[0][1] = int(sel[0][1] * hscale)
        sel[1][0] = int(sel[1][0] * wscale)
        sel[1][1] = int(sel[1][1] * hscale)
        #store bounding box
        boxCoords.append([sel[0][0], sel[0][1], sel[1][0], sel[1][1]])
    def rightClickCallback(event):
        #convert click coordinate to non-scaled image coordinates
        x = int(event.x * image.size[0]*DOWNSCALE/canvasWidth)
        y = int(event.y * image.size[1]*DOWNSCALE/canvasHeight)
        #find and remove overlapping boxes
        indices = []
        for i in range(len(boxCoords)):
            b = boxCoords[i]
            if b[0] <= x and b[1] <= y and b[2] >= x and b[3] >= y:
                indices.append(i)
                canvas.delete(boxIDs[i])
        for i in indices:
            boxCoords[i:i+1] = []
            boxIDs[i:i+1] = []
    def nextCallback(event, forward=True):
        global filenameIdx, image, imageTk, boxCoords, boxIDs
        #store box info
        filenames[filenamesSorted[filenameIdx]] = boxCoords
        #move to next file, or exit
        filenameIdx = filenameIdx+1 if forward else max(filenameIdx-1, 0)
        if filenameIdx < len(filenamesSorted):
            filename = filenamesSorted[filenameIdx]
            window.title(filename) #rename window
            canvas.delete(tkinter.ALL) #remove image and boxes
            boxIDs = []
            boxCoords = []
            #load boxes if present
            if filenames[filename] != None:
                for box in filenames[filename]:
                    boxCoords.append(box)
                    #convert to scaled image coordinates
                    wscale = canvasWidth /(image.size[0] * DOWNSCALE)
                    hscale = canvasHeight/(image.size[1] * DOWNSCALE)
                    #add box
                    boxIDs.append(
                        canvas.create_rectangle(
                            int(box[0] * wscale),
                            int(box[1] * hscale),
                            int(box[2] * wscale),
                            int(box[3] * hscale)
                        )
                    )
            #load new image
            image = Image.open(filenamesSorted[filenameIdx])
            image = image.resize((image.size[0]//DOWNSCALE, image.size[1]//DOWNSCALE), resample=Image.LANCZOS)
            imageTk = ImageTk.PhotoImage(
                image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
            )
            canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
            canvas.tag_lower(canvasImage) #move image to back
        else:
            escapeCallback(None)
    def prevCallback(event):
        nextCallback(None, False)
    def escapeCallback(event=None):
        #store box info
        if filenameIdx < len(filenamesSorted):
            filenames[filenamesSorted[filenameIdx]] = boxCoords
        #output box info
        f = sys.stdout if outputFile == None else open(outputFile, 'w')
        for filename in filenamesSorted:
            if filenames[filename] != None:
                print(filename, file=f)
                for box in filenames[filename]:
                    print(" ", end="", file=f)
                    for coord in box[:-1]:
                        print(str(coord) + ", ", end="", file=f)
                    print(box[-1], file=f)
        if not f is sys.stdout:
            f.close()
        #save images if requested
        if outputDir != None:
            for filename in filenamesSorted:
                info = filenames[filename]
                if info != None:
                    image = Image.open(filename)
                    draw = ImageDraw.Draw(image, "RGBA")
                    for box in filenames[filename]:
                        draw.rectangle([box[0], box[1], box[2], box[3]], outline=(255,0,0))
                    image.save(outputDir + "/" + os.path.basename(filename))
        sys.exit(0)
    #load boxes if present
    filename = filenamesSorted[filenameIdx]
    if filenames[filename] != None:
        for box in filenames[filename]:
            boxCoords.append(box)
            #convert to scaled image coordinates
            wscale = canvasWidth /(image.size[0] * DOWNSCALE)
            hscale = canvasHeight/(image.size[1] * DOWNSCALE)
            #add box
            boxIDs.append(
                canvas.create_rectangle(
                    int(box[0] * wscale),
                    int(box[1] * hscale),
                    int(box[2] * wscale),
                    int(box[3] * hscale)
                )
            )
    #set handlers
    canvas.bind("<Configure>", resizeCallback)
    canvas.bind("<Button-1>", clickCallback)
    canvas.bind("<B1-Motion>", moveCallback)
    canvas.bind("<ButtonRelease-1>", releaseCallback)
    canvas.bind("<Button-3>", rightClickCallback)
    window.bind("<Return>", nextCallback)
    window.bind("<Right>", nextCallback)
    window.bind("<Left>", prevCallback)
    window.bind("<Escape>", escapeCallback)
    window.protocol("WM_DELETE_WINDOW", escapeCallback)

#setup
if mode == MODE_FILTER or mode == MODE_WATER:
    #initialise cells
    cells = [
        [None for row in range(image.size[1]//INPUT_HEIGHT)]
        for col in range(image.size[0]//INPUT_WIDTH)
    ]
    #create cell outlines
    for i in range(len(cells)):
        for j in range(len(cells[0])):
            boxIDs.append(
                canvas.create_rectangle(
                    i*INPUT_WIDTH,
                    j*INPUT_HEIGHT,
                    (i+1)*INPUT_WIDTH,
                    (j+1)*INPUT_HEIGHT
                )
            )
    #setup handlers
    setupMarkCell(mode == MODE_FILTER)
elif mode == MODE_BOXES:
    setupMarkBox()

#start application
tkinter.mainloop()
