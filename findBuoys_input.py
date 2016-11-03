import math, random
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from constants import *

def getCellFilter(filterFile):
    """ Obtains filter data from 'filterFile', or uses an empty filter.
        Returns a list with the form [[0, 1, ...], ...].
            Each element denotes a row of cells, where 1 indicates a filtered cell.
    """
    if filterFile != None:
        cellFilter = []
        with open(filterFile) as file:
            for line in file:
                cellFilter.append([int(c) for c in line.strip()])
    else:
        cellFilter = [
            [0 for col in IMG_SCALED_WIDTH // INPUT_WIDTH]
            for row in IMG_SCALED_HEIGHT // INPUT_HEIGHT
        ]
    return cellFilter

#class for producing coarse network input values from a training/test data file
class CoarseBatchProducer:
    """Produces input values for the coarse network"""
    VALUES_PER_IMAGE = 300
    LOAD_IMAGES_ON_DEMAND = True
    #constructor
    def __init__(self, dataFile, cellFilter):
        self.cellFilter      = cellFilter
        self.filenames       = None #list of image files
        self.cells           = None #has the form [[[0,1,...],...],...], specifying cells in images
        self.inputs          = None
        self.outputs         = None
        self.idx             = 0
        self.valuesGenerated = 0
        #read 'dataFile'
        self.filenames = []
        cellsDict = dict()
        with open(dataFile) as file:
            filename = None
            for line in file:
                if line[0] != " ":
                    filename = line.strip()
                    self.filenames.append(filename)
                    cellsDict[filename] = []
                elif filename == None:
                    raise Exception("Invalid data file")
                else:
                    cellsDict[filename].append([int(c) for c in line.strip()])
        if len(self.filenames) == 0:
            raise Exception("No filenames")
        random.shuffle(self.filenames)
        self.cells = [cellsDict[name] for name in self.filenames]
        #allocate inputs and outputs
        self.inputs = [None for name in self.filenames]
        self.outputs = [None for name in self.filenames]
        #load images
        for i in range(1 if self.LOAD_IMAGES_ON_DEMAND else len(self.filenames)):
            self.loadImage(i)
    #load next image
    def loadImage(self, i):
        #obtain PIL image
        image = Image.open(self.filenames[i])
        image = image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #get inputs and outputs
        self.inputs[i] = []
        self.outputs[i] = []
        for row in range(len(self.cells[i])):
            for col in range(len(self.cells[i][row])):
                if self.cellFilter[row][col] == 1:
                    continue
                cellImg = image.crop(
                    (col*INPUT_WIDTH, row*INPUT_HEIGHT, (col+1)*INPUT_WIDTH, (row+1)*INPUT_HEIGHT)
                )
                #cellImg = ImageOps.autocontrast(cellImg)
                #cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
                #cellImg2 = cellImg.transpose(Image.FLIP_LEFT_RIGHT)
                #cellImg3 = cellImg.transpose(Image.FLIP_TOP_BOTTOM)
                cellImages = [
                    cellImg,  cellImg.rotate(90),  cellImg.rotate(180),  cellImg.rotate(270)
                    #, cellImg2, cellImg2.rotate(90), cellImg2.rotate(180), cellImg2.rotate(270)
                    #, cellImg3, cellImg3.rotate(90), cellImg3.rotate(180), cellImg3.rotate(270)
                ]
                data = [
                    np.array(list(img.getdata())).astype(np.float32).reshape(
                        (INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS)
                    )
                    for img in cellImages
                ]
                self.inputs[i] += data
                self.outputs[i] += [[1,0] if self.cells[i][row][col] == 1 else [0,1]] * len(cellImages)
        if len(self.inputs[i]) == 0:
            raise Exception("No unfiltered cells for \"" + self.filenames[i] + "\"")
    #returns a tuple containing a numpy array of 'size' inputs, and a numpy array of 'size' outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            if self.valuesGenerated == self.VALUES_PER_IMAGE:
                self.idx = (self.idx + 1) % len(self.inputs)
                if self.LOAD_IMAGES_ON_DEMAND and self.inputs[self.idx] == None:
                    self.loadImage(self.idx)
                self.valuesGenerated = 0
            #randomly select a non-filtered grid cell
            i = math.floor(random.random() * len(self.inputs[self.idx]))
            ##bias samples towards positive examples
            #if self.outputs[self.idx][i][0] == 0 and random.random() < 0.5:
            #    continue
            #add input and output to batch
            inputs.append(self.inputs[self.idx][i])
            outputs.append(self.outputs[self.idx][i])
            #update
            self.valuesGenerated += 1
            c += 1
        return np.array(inputs), np.array(outputs).astype(np.float32)

#class for producing detailed network input values from a training/test data file
class DetailedBatchProducer:
    """Produces input values for the detailed network"""
    VALUES_PER_IMAGE = 100
    LOAD_IMAGES_ON_DEMAND = True
    #constructor
    def __init__(self, dataFile, cellFilter):
        self.cellFilter      = cellFilter
        self.filenames       = None #list of image files
        self.boxes           = []     #has the form [[x,y,x2,y2],...], specifying boxes in images
        self.inputs          = None
        self.outputs         = None
        self.idx             = 0
        self.valuesGenerated = 0
        #read 'dataFile'
        self.filenames = []
        boxesDict = dict()
        with open(dataFile) as file:
            filename = None
            for line in file:
                if line[0] != " ":
                    filename = line.strip()
                    self.filenames.append(filename)
                    boxesDict[filename] = []
                elif filename == None:
                    raise Exception("Invalid data file")
                else:
                    boxesDict[filename].append([int(c) for c in line.strip().split(",")])
        if len(self.filenames) == 0:
            raise Exception("No filenames")
        random.shuffle(self.filenames)
        self.boxes = [boxesDict[name] for name in self.filenames]
        #allocate inputs and outputs
        self.inputs = [None for name in self.filenames]
        self.outputs = [None for name in self.filenames]
        #load images
        for i in range(1 if self.LOAD_IMAGES_ON_DEMAND else len(self.filenames)):
            self.loadImage(i)
    #load next image
    def loadImage(self, i):
        #obtain PIL image
        image = Image.open(self.filenames[i])
        image = image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #get inputs and outputs
        self.inputs[i] = []
        self.outputs[i] = []
        for row in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
            for col in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                if self.cellFilter[row][col] == 1:
                    continue
                #get cell image
                cellImg = image.crop(
                    (col*INPUT_WIDTH, row*INPUT_HEIGHT, (col+1)*INPUT_WIDTH, (row+1)*INPUT_HEIGHT)
                )
                #TODO: filter with coarse network
                #add cell image
                #cellImg = ImageOps.autocontrast(cellImg)
                #cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
                #cellImg2 = cellImg.transpose(Image.FLIP_LEFT_RIGHT)
                #cellImg3 = cellImg.transpose(Image.FLIP_TOP_BOTTOM)
                cellImages = [
                    cellImg,  cellImg.rotate(90),  cellImg.rotate(180),  cellImg.rotate(270)
                    #, cellImg2, cellImg2.rotate(90), cellImg2.rotate(180), cellImg2.rotate(270)
                    #, cellImg3, cellImg3.rotate(90), cellImg3.rotate(180), cellImg3.rotate(270)
                ]
                data = [
                    np.array(list(img.getdata())).astype(np.float32).reshape(
                        (INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS)
                    )
                    for img in cellImages
                ]
                #get inputs and outputs
                self.inputs[i] += data
                MARGIN = 10
                topLeftX = col*INPUT_WIDTH*IMG_DOWNSCALE + MARGIN
                topLeftY = row*INPUT_HEIGHT*IMG_DOWNSCALE + MARGIN
                bottomRightX = (col*INPUT_WIDTH+INPUT_WIDTH-1)*IMG_DOWNSCALE - MARGIN
                bottomRightY = (row*INPUT_HEIGHT+INPUT_HEIGHT-1)*IMG_DOWNSCALE - MARGIN
                hasOverlappingBox = False
                for box in self.boxes[self.idx]:
                    if (not box[2] < topLeftX and
                        not box[0] > bottomRightX and
                        not box[3] < topLeftY and
                        not box[1] > bottomRightY):
                        hasOverlappingBox = True
                        break
                self.outputs[i] += [[1,0] if hasOverlappingBox else [0,1]] * len(cellImages)
        if len(self.inputs[i]) == 0:
            raise Exception("No unfiltered cells for \"" + self.filenames[i] + "\"")
    #returns a tuple containing a numpy array of 'size' inputs, and a numpy array of 'size' outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            if self.valuesGenerated == self.VALUES_PER_IMAGE:
                self.idx = (self.idx + 1) % len(self.inputs)
                if self.LOAD_IMAGES_ON_DEMAND and self.inputs[self.idx] == None:
                    self.loadImage(self.idx)
                self.valuesGenerated = 0
            #randomly select a non-filtered grid cell
            i = math.floor(random.random() * len(self.inputs[self.idx]))
            ##bias samples towards positive examples
            #if self.outputs[self.idx][i][0] == 0 and random.random() < 0.5:
            #    continue
            #add input and output to batch
            inputs.append(self.inputs[self.idx][i])
            outputs.append(self.outputs[self.idx][i])
            #update
            self.valuesGenerated += 1
            c += 1
        return np.array(inputs), np.array(outputs).astype(np.float32)
