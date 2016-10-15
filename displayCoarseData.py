import sys, re
from PIL import Image, ImageTk, ImageDraw
import tkinter

usage = "Usage: python3 " + sys.argv[0] + """ [-s] file1
    Reads file1, which specifies images and marked cells, and displays them.
    file1 should have the same format as output by genCoarseData.py.
    Pressing enter causes the next image to be displayed.

    Options:
        -s
            Save images to files. A file 'images/img1.jpg' is saved as 'images/img1_marked.jpg'.
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
images = [] #images[i] has the form [filename, [rowString, ...]]
imageIdx = 0
with open(dataFile) as file:
    for line in file:
        if line[0] != " ":
            images.append([line.strip(), []])
        else:
            images[-1][1].append([int(c) for c in line.strip()])
if len(images) == 0:
    sys.exit(0)
filename = images[0][0]

#create window
window = tkinter.Tk()
window.title(filename)
image = Image.open(filename)
draw = ImageDraw.Draw(image, "RGBA")
canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
canvas.pack()
imageTk = ImageTk.PhotoImage(image)
canvas.create_image(image.size[0]//2, image.size[1]//2, image=imageTk)

#variables
IMG_DOWNSCALE = 2
INPUT_HEIGHT = 32*IMG_DOWNSCALE
INPUT_WIDTH  = 32*IMG_DOWNSCALE

#helper functions
def drawBoxes():
    imageInfo = images[imageIdx]
    for i in range(image.size[0]//INPUT_WIDTH):
        for j in range(image.size[1]//INPUT_HEIGHT):
            topLeftX = i*INPUT_WIDTH
            topLeftY = j*INPUT_HEIGHT
            bottomRightX = i*INPUT_WIDTH+INPUT_WIDTH-1
            bottomRightY = j*INPUT_HEIGHT+INPUT_HEIGHT-1
            #draw grid box
            canvas.create_rectangle(topLeftX, topLeftY, bottomRightX, bottomRightY)
            draw.rectangle(
                [topLeftX, topLeftY, bottomRightX, bottomRightY],
                outline=(0,0,0)
            )
            #draw marking
            flag = imageInfo[1][j][i]
            if flag == 1:
                canvas.create_rectangle(
                    topLeftX, topLeftY, bottomRightX, bottomRightY, fill="green", stipple='gray50'
                )
                draw.rectangle(
                    [topLeftX, topLeftY, bottomRightX, bottomRightY],
                    fill=(0,128,0,128)
                )
            elif flag == 2:
                canvas.create_rectangle(
                    topLeftX, topLeftY, bottomRightX, bottomRightY, fill="orange", stipple='gray50'
                )
                draw.rectangle(
                    [topLeftX, topLeftY, bottomRightX, bottomRightY],
                    fill=(128,128,0,128)
                )

#draw boxes for current image file
drawBoxes()

#define enter key handler
def returnCallback(event):
    global filename, imageIdx, image, draw, imageTk
    #save file if requested
    if saveFiles:
        image.save(re.sub(r'(\.[^.]*)?$', r'_marked\1', filename))
    #move to next image, or exit
    imageIdx += 1
    if imageIdx < len(images):
        #remove image and boxes
        canvas.delete(tkinter.ALL)
        #load new image
        filename = images[imageIdx][0]
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
