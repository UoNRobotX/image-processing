import sys, os, random
from PIL import Image, ImageTk, ImageDraw
import tkinter

from .constants import *

class Window:
    #constructor
    def __init__(self, mode, skipFile, cellFilter, fileMarks, outputFile, saveDir):
        self.mode       = mode
        self.cellFilter = cellFilter
        self.fileMarks  = fileMarks
        self.outputFile = outputFile
        self.saveDir    = saveDir
        #fields used when traversing the file list
        self.fileIdx   = 0
        self.filenames = [name for name in fileMarks]
        self.filenames.sort()
        #fields used to hold mark data
        self.sel           = [[0,0], [0,0]] #corners of a box created while dragging the mouse
        self.box           = None   #holds the ID of a rectangle shown while dragging the mouse
        self.boxCoords     = []     #[[x,y,x,y,t], ...], describes created boxes
        self.boxIDs        = []     #holds IDs of elements in "boxCoords"
        self.cells         = []     #cells[col][row] is an ID or None, indicating cell marked-ness
        self.mouseDownCell = [0, 0] #specifies the last cell the mouse was in while held down
        self.boxType       = 0      #the current box type being marked
        #skip to a file if requested
        if skipFile != None:
            found = False
            for i in range(len(self.filenames)):
                if os.path.basename(self.filenames[i]) == skipFile:
                    self.fileIdx = i
                    found = True
            if not found:
                raise Exception("Skip file not found")
        #create window
        self.window = tkinter.Tk()
        self.window.title(self.filenames[self.fileIdx])
        #obtain image
        self.image = Image.open(self.filenames[self.fileIdx])
        checkImage(self.image, self.filenames[self.fileIdx])
        #create canvas, and add the image to it
        self.canvasWidth = IMG_WIDTH
        self.canvasHeight = IMG_HEIGHT
        self.canvas = tkinter.Canvas(
            self.window, width=self.canvasWidth, height=self.canvasHeight
        )
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
        self.imageTk = ImageTk.PhotoImage(self.image)
        self.canvasImage = self.canvas.create_image(
            self.canvasWidth//2, self.canvasHeight//2, image=self.imageTk
        )
        #setup
        if mode == "filter" or mode == "coarse":
            #initialise cells
            self.cells = [
                [None for row in range(IMG_HEIGHT//CELL_HEIGHT)]
                for col in range(IMG_WIDTH//CELL_WIDTH)
            ]
            #create cell outlines
            for i in range(len(self.cells)):
                for j in range(len(self.cells[i])):
                    self.boxIDs.append(
                        self.canvas.create_rectangle(
                            i*CELL_WIDTH,
                            j*CELL_HEIGHT,
                            (i+1)*CELL_WIDTH,
                            (j+1)*CELL_HEIGHT
                        )
                    )
            #setup handlers
            if mode == "filter":
                self.setupMarkFilter()
            else:
                self.setupMarkCoarse()
        elif mode == "detailed":
            self.setupMarkDetailed()
        elif mode == "windows":
            self.setupShowWindows()
        #start application
        tkinter.mainloop()
    #setup functions
    def setupMarkFilter(self):
        #load cell filter if provided
        if self.cellFilter != None:
            for row in range(len(self.cellFilter)):
                for col in range(len(self.cellFilter[0])):
                    if self.cellFilter[row][col] == 1:
                        self.toggleCell(col, row)
        #set handlers
        self.canvas.bind("<Configure>", self.resizeCallback)
        self.canvas.bind("<Button-1>",  self.markCellClickCallback)
        self.canvas.bind("<B1-Motion>", self.markCellMoveCallback)
        self.window.bind("<Right>",     self.markFilterNextCallback)
        self.window.bind("<Left>",      self.markFilterPrevCallback)
        self.window.bind("<Escape>",    self.markFilterEscapeCallback)
        self.window.protocol("WM_DELETE_WINDOW", self.markFilterEscapeCallback)
    def setupMarkCoarse(self):
        #load water cells if provided
        if self.fileMarks[self.filenames[self.fileIdx]] != None:
            info = self.fileMarks[self.filenames[self.fileIdx]]
            for row in range(len(info)):
                for col in range(len(info[0])):
                    if info[row][col] == 1:
                        self.toggleCell(col, row)
        #set handlers
        self.canvas.bind("<Configure>", self.resizeCallback)
        self.canvas.bind("<Button-1>",  self.markCellClickCallback)
        self.canvas.bind("<B1-Motion>", self.markCellMoveCallback)
        self.window.bind("<Right>",     self.markCoarseNextCallback)
        self.window.bind("<Left>",      self.markCoarsePrevCallback)
        self.window.bind("<Escape>",    self.markCoarseEscapeCallback)
        self.window.protocol("WM_DELETE_WINDOW", self.markCoarseEscapeCallback)
    def setupMarkDetailed(self):
        self.setTitleWithMode()
        filename = self.filenames[self.fileIdx]
        #load boxes if present
        if self.fileMarks[filename] != None:
            for box in self.fileMarks[filename]:
                self.boxCoords.append(box)
                self.boxIDs.append(self.canvas.create_rectangle(
                    box[0], box[1], box[2], box[3],
                    outline=BOX_COLORS[box[4]],
                    width=2
                ))
        #set handlers
        self.canvas.bind("<Configure>",       self.resizeCallback)
        self.canvas.bind("<Button-1>",        self.markDetailedClickCallback)
        self.canvas.bind("<B1-Motion>",       self.markDetailedMoveCallback)
        self.canvas.bind("<ButtonRelease-1>", self.markDetailedReleaseCallback)
        self.canvas.bind("<Button-3>",        self.markDetailedRightClickCallback)
        self.window.bind("<Right>",           self.markDetailedNextCallback)
        self.window.bind("<Left>",            self.markDetailedPrevCallback)
        self.window.bind("<Escape>",          self.markDetailedEscapeCallback)
        self.window.bind("<Tab>",             self.markDetailedTabCallback)
        for i in range(min(9, NUM_BOX_TYPES)):
            self.window.bind(str(i+1), self.markDetailedDigitCallback)
        self.window.protocol("WM_DELETE_WINDOW", self.markDetailedEscapeCallback)
    def setupShowWindows(self):
        colors = ["red", "green", "blue", "yellow", "pink", "brown", "black"]
        if True: #draw many cells
            cells = GET_WINDOWS()
            for i in range(len(cells)):
                if random.random() > 0.05: continue #randomly skip
                #get cell position
                cell = cells[i]
                self.canvas.create_rectangle(
                    cell[0]+2, cell[1]+2, cell[2], cell[3],
                    outline=colors[i % len(colors)], width=2
                )
        else: #draw some cells
            for i in range(len(WINDOW_SCALES)):
                scale = WINDOW_SCALES[i]
                cellHeight = int(INPUT_HEIGHT * scale)
                cellWidth  = int(INPUT_WIDTH * scale)
                x = int(i * IMG_WIDTH / len(WINDOW_SCALES))
                cell = [x, WINDOW_MAX_Y[i] - cellHeight, x + cellWidth, WINDOW_MAX_Y[i]]
                while True:
                    self.canvas.create_rectangle(
                        cell[0], cell[1], cell[2], cell[3], outline=colors[i], width=2
                    )
                    #move right and up
                    cell[0] += int(cellWidth * WINDOW_STEP_X)
                    cell[2] += int(cellWidth * WINDOW_STEP_X)
                    cell[1] -= int(cellHeight * WINDOW_STEP_Y)
                    cell[3] -= int(cellHeight * WINDOW_STEP_Y)
                    if cell[2] > IMG_WIDTH or cell[1] < WINDOW_MIN_Y[i]:
                        break
        #set handlers
        self.canvas.bind("<Configure>", self.resizeCallback)
        self.window.bind("<Return>",    self.markFilterNextCallback)
        self.window.bind("<Right>",     self.markFilterNextCallback)
        self.window.bind("<Left>",      self.markFilterPrevCallback)
        self.window.bind("<Escape>",    self.showWindowsEscapeCallback)
        self.window.protocol("WM_DELETE_WINDOW", self.showWindowsEscapeCallback)
    #callback functions
    def resizeCallback(self, event):
        wscale = event.width /self.canvasWidth
        hscale = event.height/self.canvasHeight
        self.canvasWidth  = event.width
        self.canvasHeight = event.height
        self.canvas.scale("all", 0, 0, wscale, hscale)
        self.canvas.delete(self.canvasImage)
        self.imageTk = ImageTk.PhotoImage(
            self.image.resize((self.canvasWidth, self.canvasHeight), resample=Image.LANCZOS)
        )
        self.canvasImage = self.canvas.create_image(
            self.canvasWidth//2, self.canvasHeight//2, image=self.imageTk
        )
        self.canvas.tag_lower(self.canvasImage) #move image to back
    def markCellClickCallback(self, event):
        i = int(event.x / self.canvasWidth  * IMG_WIDTH)  // CELL_WIDTH
        j = int(event.y / self.canvasHeight * IMG_HEIGHT) // CELL_HEIGHT
        self.mouseDownCell = [i, j]
        self.toggleCell(i,j)
    def markCellMoveCallback(self, event):
        #do nothing if mouse is outside window
        if not 0 <= event.x < self.canvasWidth or not 0 <= event.y < self.canvasHeight:
            return
        #if a new cell was moved to, toggle it
        i = int(event.x / self.canvasWidth  * IMG_WIDTH)  // CELL_WIDTH
        j = int(event.y / self.canvasHeight * IMG_HEIGHT) // CELL_HEIGHT
        if i != self.mouseDownCell[0] or j != self.mouseDownCell[1]:
            self.mouseDownCell = [i, j]
            self.toggleCell(i,j)
    def markFilterNextCallback(self, event, forward=True):
        #do nothing if at first or last file
        if not forward and self.fileIdx == 0 or forward and self.fileIdx == len(self.filenames) -1:
            return
        #move to next file, or exit
        self.fileIdx = self.fileIdx + 1 if forward else self.fileIdx - 1
        filename = self.filenames[self.fileIdx]
        self.window.title(filename)
        #load new image
        self.canvas.delete(self.canvasImage)
        self.image = Image.open(filename)
        checkImage(self.image, filename)
        self.imageTk = ImageTk.PhotoImage(
            self.image.resize((self.canvasWidth, self.canvasHeight), resample=Image.LANCZOS)
        )
        self.canvasImage = self.canvas.create_image(
            self.canvasWidth//2, self.canvasHeight//2, image=self.imageTk
        )
        self.canvas.tag_lower(self.canvasImage) #move image to back
    def markFilterPrevCallback(self, event):
        self.markFilterNextCallback(None, forward=False)
    def markFilterEscapeCallback(self, event=None):
        #output filter info
        f = sys.stdout if self.outputFile == None else open(self.outputFile, "w")
        for row in range(len(self.cells[0])):
            for col in range(len(self.cells)):
                print("0" if self.cells[col][row] == None else "1", end="", file=f)
            print(file=f)
        if not f is sys.stdout:
            f.close()
        #save images if requested
        if self.saveDir != None:
            for filename in self.filenames:
                image = Image.open(filename)
                draw = ImageDraw.Draw(image, "RGBA")
                for i in range(IMG_WIDTH//CELL_WIDTH):
                    for j in range(IMG_HEIGHT//CELL_HEIGHT):
                        topLeftX = i * CELL_WIDTH
                        topLeftY = j * CELL_HEIGHT
                        bottomRightX = (i+1) * CELL_WIDTH - 1
                        bottomRightY = (j+1) * CELL_HEIGHT - 1
                        #draw grid box
                        draw.rectangle(
                            [topLeftX, topLeftY, bottomRightX, bottomRightY],
                            outline=(0,0,0)
                        )
                        #draw marking
                        if self.cells[i][j] != None:
                            draw.rectangle(
                                [topLeftX, topLeftY, bottomRightX, bottomRightY],
                                fill=(0,128,0,128)
                            )
                image.save(self.saveDir + "/" + os.path.basename(filename))
        sys.exit(0)
    def markCoarseNextCallback(self, event, forward=True):
        #do nothing if at first or last file
        if not forward and self.fileIdx == 0 or forward and self.fileIdx == len(self.filenames) -1:
            return
        #move to next file
        self.fileIdx = self.fileIdx + 1 if forward else self.fileIdx - 1
        #store mark info
        self.fileMarks[self.filenames[self.fileIdx]] = [
            [0 if self.cells[col][row] == None else 1 for col in range(len(self.cells))]
            for row in range(len(self.cells[0]))
        ]
        #move to next file
        self.fileIdx = self.fileIdx + 1 if forward else self.fileIdx - 1
        filename = self.filenames[self.fileIdx]
        self.window.title(filename) #rename window
        #remove colored boxes
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                self.canvas.delete(self.cells[i][j])
                self.cells[i][j] = None
        #load colored boxes if present
        if self.fileMarks[filename] != None:
            info = self.fileMarks[filename]
            for row in range(len(info)):
                for col in range(len(info[0])):
                    if info[row][col] == 1:
                        self.toggleCell(col, row)
        #load new image
        self.canvas.delete(self.canvasImage)
        self.image = Image.open(filename)
        checkImage(self.image, self.filenames[self.fileIdx])
        self.imageTk = ImageTk.PhotoImage(
            self.image.resize((self.canvasWidth, self.canvasHeight), resample=Image.LANCZOS)
        )
        self.canvasImage = self.canvas.create_image(
            self.canvasWidth//2, self.canvasHeight//2, image=self.imageTk
        )
        self.canvas.tag_lower(self.canvasImage) #move image to back
    def markCoarsePrevCallback(self, event):
        self.markCoarseNextCallback(None, forward=False)
    def markCoarseEscapeCallback(self, event=None):
        #store info
        self.fileMarks[self.filenames[self.fileIdx]] = [
            [0 if self.cells[col][row] == None else 1 for col in range(len(self.cells))]
            for row in range(len(self.cells[0]))
        ]
        #output info
        f = sys.stdout if self.outputFile == None else open(self.outputFile, "w")
        for filename in self.filenames:
            if self.fileMarks[filename] != None:
                print(filename, file=f)
                info = self.fileMarks[filename]
                for row in range(len(info)):
                    print(" ", end="", file=f)
                    for col in range(len(info[row])):
                        print("0" if info[row][col] == 0 else "1", end="", file=f)
                    print(file=f)
        if not f is sys.stdout:
            f.close()
        #save images if requested
        if self.saveDir != None:
            for filename in self.filenames:
                info = self.fileMarks[filename]
                if info != None:
                    image = Image.open(filename)
                    draw = ImageDraw.Draw(image, "RGBA")
                    for i in range(IMG_WIDTH//CELL_WIDTH):
                        for j in range(IMG_HEIGHT//CELL_HEIGHT):
                            topLeftX = i * CELL_WIDTH
                            topLeftY = j * CELL_HEIGHT
                            bottomRightX = (i+1) * CELL_WIDTH - 1
                            bottomRightY = (j+1) * CELL_HEIGHT - 1
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
                    image.save(self.saveDir + "/" + os.path.basename(filename))
        sys.exit(0)
    def markDetailedClickCallback(self, event):
        self.sel[0] = [event.x, event.y]
    def markDetailedMoveCallback(self, event):
        self.sel[1] = [event.x, event.y]
        self.canvas.delete(self.box)
        self.box = self.canvas.create_rectangle(
            self.sel[0][0], self.sel[0][1], self.sel[1][0], self.sel[1][1],
            outline=BOX_COLORS[self.boxType],
            width=2
        )
    def markDetailedReleaseCallback(self, event):
        self.sel[1] = [event.x, event.y]
        self.canvas.delete(self.box)
        #ignore if box is too small
        MIN_SIZE = 5
        if abs(self.sel[0][0] - self.sel[1][0]) < MIN_SIZE or \
            abs(self.sel[0][1] - self.sel[1][1]) < MIN_SIZE:
            return
        #make "sel" contain top-left and bottom-right
        if self.sel[0][0] > self.sel[1][0]:
            temp = self.sel[0][0]
            self.sel[0][0] = self.sel[1][0]
            self.sel[1][0] = temp
        if self.sel[0][1] > self.sel[1][1]:
            temp = self.sel[0][1]
            self.sel[0][1] = self.sel[1][1]
            self.sel[1][1] = temp
        #add box
        self.boxIDs.append(
            self.canvas.create_rectangle(
                self.sel[0][0], self.sel[0][1], self.sel[1][0], self.sel[1][1], \
                outline=BOX_COLORS[self.boxType],
                width=2
            )
        )
        #convert to non-scaled image coordinates
        wscale = IMG_WIDTH/self.canvasWidth
        hscale = IMG_HEIGHT/self.canvasHeight
        self.sel[0][0] = int(self.sel[0][0] * wscale)
        self.sel[0][1] = int(self.sel[0][1] * hscale)
        self.sel[1][0] = int(self.sel[1][0] * wscale)
        self.sel[1][1] = int(self.sel[1][1] * hscale)
        #store bounding box
        self.boxCoords.append(
            [self.sel[0][0], self.sel[0][1], self.sel[1][0], self.sel[1][1], self.boxType]
        )
    def markDetailedRightClickCallback(self, event):
        #convert click coordinate to non-scaled image coordinates
        x = int(event.x / self.canvasWidth * IMG_WIDTH)
        y = int(event.y / self.canvasHeight * IMG_HEIGHT)
        #find and remove overlapping boxes
        indices = []
        for i in range(len(self.boxCoords)):
            b = self.boxCoords[i]
            if b[0] <= x and b[1] <= y and b[2] >= x and b[3] >= y:
                indices.append(i)
                self.canvas.delete(self.boxIDs[i])
        for i in indices:
            self.boxCoords[i:i+1] = []
            self.boxIDs[i:i+1] = []
    def markDetailedNextCallback(self, event, forward=True):
        #do nothing if moving before first or after last file
        if not forward and self.fileIdx == 0 or forward and self.fileIdx == len(self.filenames) -1:
            return
        #store box info
        self.fileMarks[self.filenames[self.fileIdx]] = self.boxCoords
        #move to next file
        self.fileIdx = self.fileIdx + 1 if forward else self.fileIdx - 1
        filename = self.filenames[self.fileIdx]
        self.setTitleWithMode()
        self.canvas.delete(tkinter.ALL) #remove image and boxes
        self.boxIDs = []
        self.boxCoords = []
        #load boxes if present
        if self.fileMarks[filename] != None:
            for box in self.fileMarks[filename]:
                self.boxCoords.append(box)
                self.boxIDs.append(
                    self.canvas.create_rectangle(
                        int(box[0] / IMG_WIDTH  * self.canvasWidth),
                        int(box[1] / IMG_HEIGHT * self.canvasHeight),
                        int(box[2] / IMG_WIDTH  * self.canvasWidth),
                        int(box[3] / IMG_HEIGHT * self.canvasHeight),
                        outline=BOX_COLORS[box[4]],
                        width=2
                    )
                )
        #load new image
        self.image = Image.open(filename)
        checkImage(self.image, filename)
        self.imageTk = ImageTk.PhotoImage(
            self.image.resize((self.canvasWidth, self.canvasHeight), resample=Image.LANCZOS)
        )
        self.canvasImage = self.canvas.create_image(
            self.canvasWidth//2, self.canvasHeight//2, image=self.imageTk
        )
        self.canvas.tag_lower(self.canvasImage) #move image to back
    def markDetailedPrevCallback(self, event):
        self.markDetailedNextCallback(None, False)
    def markDetailedEscapeCallback(self, event=None):
        #store box info
        self.fileMarks[self.filenames[self.fileIdx]] = self.boxCoords
        #output box info
        f = sys.stdout if self.outputFile == None else open(self.outputFile, "w")
        for filename in self.filenames:
            if self.fileMarks[filename] != None:
                print(filename, file=f)
                for box in self.fileMarks[filename]:
                    print(" " + ",".join([str(val) for val in box]), file=f)
        if not f is sys.stdout:
            f.close()
        #save images if requested
        if self.saveDir != None:
            for filename in self.filenames:
                info = self.fileMarks[filename]
                if info != None:
                    image = Image.open(filename)
                    draw = ImageDraw.Draw(image, "RGBA")
                    for box in self.fileMarks[filename]:
                        draw.rectangle(
                            [box[0], box[1], box[2], box[3]],
                            outline=BOX_COLORS[box[4]],
                            width=2
                        )
                    image.save(self.saveDir + "/" + os.path.basename(filename))
        sys.exit(0)
    def markDetailedTabCallback(self, event):
        self.boxType = (self.boxType + 1) % NUM_BOX_TYPES
        self.setTitleWithMode()
    def markDetailedDigitCallback(self, event):
        self.boxType = int(str(event.char))-1
        self.setTitleWithMode()
    def showWindowsEscapeCallback(self, event=None):
        if self.saveDir != None:
            raise Exception("Saving images is not implemented")
        sys.exit(0)
    #helper functions
    def toggleCell(self, i, j): #toggle marked-ness of cell i-j
        if self.cells[i][j] != None:
            self.canvas.delete(self.cells[i][j])
            self.cells[i][j] = None
        else:
            wscale = self.canvasWidth  / IMG_WIDTH
            hscale = self.canvasHeight / IMG_HEIGHT
            self.cells[i][j] = self.canvas.create_rectangle(
                i * CELL_WIDTH  * wscale,
                j * CELL_HEIGHT * hscale,
                (i + 1) * CELL_WIDTH  * wscale,
                (j + 1) * CELL_HEIGHT * hscale,
                fill="green",
                stipple="gray50"
            )
    def setTitleWithMode(self):
        filename = self.filenames[self.fileIdx]
        self.window.title(filename + ", type " + str(self.boxType+1) + "/" + str(NUM_BOX_TYPES))

def checkImage(image, filename):
    if image.size[0] != IMG_WIDTH or image.size[1] != IMG_HEIGHT:
        raise Exception("Unexpected size for image \"" + filename + "\"")
    if IMG_CHANNELS != 3:
        raise Exception("An IMG_CHANNELS other than 3 has not been implemented")
    if image.mode != "RGB":
        raise Exception("Unexpected mode for image \"" + filename + "\"")
