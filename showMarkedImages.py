import sys, re, os
from PIL import Image, ImageTk, ImageDraw
import tkinter

usage = "Usage: python3 " + sys.argv[0] + """ [-f file1 imageOrDir1] [-w file1] [-b file1] [-s]
    Reads file1, which specifies image information, and displays it.
    Pressing enter causes the next image to be displayed.

    At least one of -f, -w, or -b should be given.
    If more than one is given, only the last is used.

    Options:
        -f file1 imageOrDir1
            Display a filter grid specified by 'file1', over images specified by 'imageOrDir1'.
            'file1' should have the format output by using 'markImages.py' with -f.
        -w file1
            Display water cells specified by 'file1'.
            'file1' should have the format output by using 'markImages.py' with -w.
        -b file1
            Display boxes specified by 'file1'.
            'file1' should have the format output by using 'markImages.py' with -b.
        -s
            Save images to files. A file 'images/img1.jpg' is saved as 'images/img1_out.jpg'.
"""

#process command line arguments
MODE_BOXES  = 0
MODE_FILTER = 1
MODE_WATER  = 2
mode        = None
dataFile    = None  #the file to read info from
imageOrDir  = None  #with -f, the specified image or directory of images
saveFiles   = False #True if image files are to be saved
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "-b":
        mode = MODE_BOXES
        i += 1
        if i < len(sys.argv):
            dataFile = sys.argv[i]
        i += 1
        if i < len(sys.argv):
            imageOrDir = sys.argv[i]
    elif arg == "-f":
        mode = MODE_FILTER
        i += 1
        if i < len(sys.argv):
            dataFile = sys.argv[i]
        i += 1
        if i < len(sys.argv):
            imageOrDir = sys.argv[i]
    elif arg == "-w":
        mode = MODE_WATER
        i += 1
        if i < len(sys.argv):
            dataFile = sys.argv[i]
    elif arg == "-s":
        saveFiles = True
    else:
        print(usage, file=sys.stderr)
        sys.exit(1)
    i += 1
if mode == None:
    print("At least one of -f, -w, or -b should be given", file=sys.stderr)
    sys.exit(1)
if dataFile == None or mode == MODE_FILTER and imageOrDir == None:
    print(usage, file=sys.stderr)
    sys.exit(1)

#read data file
filenames = [] #image files to display
records   = [] #contains info for each image file
fileIdx   = 0
with open(dataFile) as file:
    if mode == MODE_BOXES:
        filenameSet = set()
        boxDict = dict()
        for line in file:
            if line[0] != " ":
                filenames.append(line.strip())
                records.append([])
            else:
                records[-1].append([int(c) for c in line.strip().split(",")])
    elif mode == MODE_FILTER:
        if os.path.isfile(imageOrDir):
            filenames = [imageOrDir]
        elif os.path.isdir(imageOrDir):
            filenames = [
                imageOrDir + "/" + name for
                name in os.listdir(imageOrDir) if
                os.path.isfile(imageOrDir + "/" + name) and re.fullmatch(r".*\.jpg", name)
            ]
            filenames.sort()
        else:
            print("Invalid imageOrDir1 argument to -b", file=sys.stderr)
            sys.exit(1)
        for line in file:
            records.append([int(c) for c in line.strip()])
    elif mode == MODE_WATER:
        for line in file:
            if line[0] != " ":
                filenames.append(line.strip())
                records.append([])
            else:
                records[-1].append([int(c) for c in line.strip()])
if len(filenames) == 0:
    sys.exit(0)

#create window
window = tkinter.Tk()
window.title(filenames[0])
image = Image.open(filenames[0])
draw = ImageDraw.Draw(image, "RGBA")
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

#helper functions
def drawMarks(mode):
    if mode == MODE_BOXES:
        for box in records[fileIdx]:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=(255,0,0))
    else:
        cells = records if mode == MODE_FILTER else records[fileIdx]
        for i in range(image.size[0]//INPUT_WIDTH):
            for j in range(image.size[1]//INPUT_HEIGHT):
                topLeftX = i*INPUT_WIDTH
                topLeftY = j*INPUT_HEIGHT
                bottomRightX = i*INPUT_WIDTH+INPUT_WIDTH-1
                bottomRightY = j*INPUT_HEIGHT+INPUT_HEIGHT-1
                #draw grid box
                draw.rectangle(
                    [topLeftX, topLeftY, bottomRightX, bottomRightY],
                    outline=(0,0,0)
                )
                #draw marking
                if cells[j][i] == 1:
                    draw.rectangle(
                        [topLeftX, topLeftY, bottomRightX, bottomRightY],
                        fill=(0,128,0,128)
                    )

#draw boxes/cells for current image file
drawMarks(mode)

#handler functions
def resizeCallback(event):
    global canvasWidth, canvasHeight, imageTk, canvasImage
    canvasWidth = event.width
    canvasHeight = event.height
    canvas.config(width=canvasWidth, height=canvasHeight)
    canvas.delete(canvasImage)
    imageTk = ImageTk.PhotoImage(
        image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
    )
    canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
def returnCallback(event):
    global fileIdx, image, draw, imageTk, canvasImage
    #save file if requested
    if saveFiles:
        image.save(re.sub(r"(\.[^.]*)?$", r"_out\1", filenames[fileIdx]))
    #move to next image, or exit
    fileIdx += 1
    if fileIdx < len(filenames):
        #remove image
        canvas.delete(canvasImage)
        #load new image
        window.title(filenames[fileIdx])
        image = Image.open(filenames[fileIdx])
        draw = ImageDraw.Draw(image, "RGBA")
        #draw marks, and scale them
        drawMarks(mode)
        #add scaled image
        imageTk = ImageTk.PhotoImage(
            image.resize((canvasWidth, canvasHeight), resample=Image.LANCZOS)
        )
        canvasImage = canvas.create_image(canvasWidth//2, canvasHeight//2, image=imageTk)
    else:
        sys.exit(0)
def escapeCallback(event):
    sys.exit(0)

#bind handlers
canvas.bind("<Configure>", resizeCallback)
window.bind("<Return>", returnCallback)
window.bind("<Escape>", escapeCallback)

#start application
tkinter.mainloop()
