import math, random
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .constants import *

def getCellFilter(filterFile):
    """ Obtains filter data from "filterFile", or uses an empty filter.
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

class CoarseBatchProducer:
    """Produces input values for the coarse network"""
    VALUES_PER_IMAGE = 300
    #constructor
    def __init__(self, dataFile, cellFilter):
        self.cellFilter      = cellFilter
        self.filenames       = None #list of image files
        self.cells           = None #has the form [[[0,1,...],...],...], specifying cells in images
        self.inputs          = None
        self.outputs         = None
        self.idx             = 0
        self.valuesGenerated = 0
        #read "dataFile"
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
        for i in range(1 if LOAD_IMAGES_ON_DEMAND else len(self.filenames)):
            self.loadImage(i)
    #load next image
    def loadImage(self, fileIdx):
        #obtain PIL image
        image = Image.open(self.filenames[fileIdx])
        image = image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #get inputs and outputs
        self.inputs[fileIdx] = []
        self.outputs[fileIdx] = []
        for row in range(len(self.cells[fileIdx])):
            for col in range(len(self.cells[fileIdx][row])):
                #use static filter
                if self.cellFilter[row][col] == 1:
                    continue
                #get cell image
                cellImg = image.crop(
                    (col*INPUT_WIDTH, row*INPUT_HEIGHT, (col+1)*INPUT_WIDTH, (row+1)*INPUT_HEIGHT)
                )
                #preprocess image
                if False: #maximise image contrast
                    cellImg = ImageOps.autocontrast(cellImg)
                if False: #blur image
                    cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
                cellImages = [cellImg]
                if True: #add rotated images
                    cellImages += [cellImg.rotate(180) for img in cellImages]
                    cellImages += [cellImg.rotate(90) for img in cellImages]
                if False: #add flip images
                    cellImages += [cellImg.transpose(Image.FLIP_LEFT_RIGHT) for img in cellImages]
                if False: #add sheared images
                    shearFactor = random.random()*0.8 - 0.4
                    cellImages += [
                        img.transform(
                            (img.size[0], img.size[1]),
                            Image.AFFINE,
                            data=(
                                (1-shearFactor, shearFactor, 0, 0, 1, 0) if shearFactor>0 else
                                (1+shearFactor, shearFactor, -shearFactor*img.size[0], 0, 1, 0)
                            ),
                            resample=Image.BICUBIC)
                        for img in cellImages
                    ]
                #get inputs and outputs
                data = [
                    np.array(list(img.getdata())).astype(np.float32).reshape(
                        (INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS)
                    )
                    for img in cellImages
                ]
                self.inputs[fileIdx] += data
                containsWater = self.cells[fileIdx][row][col] == 1
                self.outputs[fileIdx] += [[1,0] if containsWater else [0,1]] * len(cellImages)
        if len(self.inputs[fileIdx]) == 0:
            raise Exception("No unfiltered cells for \"" + self.filenames[fileIdx] + "\"")
    #returns a tuple containing a numpy array of "size" inputs, and a numpy array of "size" outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            if self.valuesGenerated == self.VALUES_PER_IMAGE:
                self.idx = (self.idx + 1) % len(self.inputs)
                if LOAD_IMAGES_ON_DEMAND and self.inputs[self.idx] == None:
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

class DetailedBatchProducer:
    """Produces input values for the detailed network"""
    VALUES_PER_IMAGE = 400
    #constructor
    def __init__(self, dataFile, cellFilter):
        self.cellFilter      = cellFilter
        self.filenames       = None #list of image files
        self.boxes           = []     #has the form [[x,y,x2,y2],...], specifying boxes in images
        self.inputs          = None
        self.outputs         = None
        self.idx             = 0
        self.valuesGenerated = 0
        #read "dataFile"
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
        #random.shuffle(self.filenames)
        self.boxes = [boxesDict[name] for name in self.filenames]
        #allocate inputs and outputs
        self.inputs = [None for name in self.filenames]
        self.outputs = [None for name in self.filenames]
        #load images
        for i in range(1 if LOAD_IMAGES_ON_DEMAND else len(self.filenames)):
            self.loadImage(i)
    #load next image
    def loadImage(self, fileIdx):
        #obtain PIL image
        image = Image.open(self.filenames[fileIdx])
        image = image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #get inputs and outputs
        self.inputs[fileIdx] = []
        self.outputs[fileIdx] = []
        stride = (INPUT_WIDTH//2, INPUT_HEIGHT//2)
        numHorizontalSteps = (IMG_SCALED_WIDTH  // stride[0]) - INPUT_WIDTH  // stride[0]
        numVerticalSteps   = (IMG_SCALED_HEIGHT // stride[1]) - INPUT_HEIGHT // stride[1]
        for i in range(numVerticalSteps):
            for j in range(numHorizontalSteps):
                #get cell position
                x = j*stride[0]
                y = i*stride[1]
                #use static filter
                intersectingCols = [x // INPUT_WIDTH]
                intersectingRows = [y // INPUT_HEIGHT]
                if x % INPUT_WIDTH != 0:
                    intersectingCols.append(x // INPUT_WIDTH + 1)
                if y % INPUT_HEIGHT != 0:
                    intersectingRows.append(y // INPUT_HEIGHT + 1)
                intersectingCells = (
                    (row, col) for row in intersectingRows for col in intersectingCols
                )
                hasOverlappingFilteredCell = False
                for (row, col) in intersectingCells:
                    if self.cellFilter[row][col] == 1:
                        hasOverlappingFilteredCell = True
                        break
                if hasOverlappingFilteredCell:
                    continue
                #get cell image
                cellImg = image.crop(
                    (x, y, x+INPUT_WIDTH, y+INPUT_HEIGHT)
                )
                #TODO: filter with coarse network?
                #preprocess image
                if False: #maximise image contrast
                    cellImg = ImageOps.autocontrast(cellImg)
                if False: #blur image
                    cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
                cellImages = [cellImg]
                if True: #add rotated images
                    cellImages += [cellImg.rotate(180) for img in cellImages]
                    cellImages += [cellImg.rotate(90) for img in cellImages]
                if False: #add flip images
                    cellImages += [cellImg.transpose(Image.FLIP_LEFT_RIGHT) for img in cellImages]
                if False: #add sheared images
                    shearFactor = random.random()*0.8 - 0.4
                    cellImages += [
                        img.transform(
                            (img.size[0], img.size[1]),
                            Image.AFFINE,
                            data=(
                                (1-shearFactor, shearFactor, 0, 0, 1, 0) if shearFactor>0 else
                                (1+shearFactor, shearFactor, -shearFactor*img.size[0], 0, 1, 0)
                            ),
                            resample=Image.BICUBIC)
                        for img in cellImages
                    ]
                #get input
                data = [
                    np.array(list(img.getdata())).astype(np.float32).reshape(
                        (INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS)
                    )
                    for img in cellImages
                ]
                self.inputs[fileIdx] += data
                #get output
                topLeftX = x*IMG_DOWNSCALE
                topLeftY = y*IMG_DOWNSCALE
                bottomRightX = (x+INPUT_WIDTH-1)*IMG_DOWNSCALE
                bottomRightY = (y+INPUT_HEIGHT-1)*IMG_DOWNSCALE
                hasOverlappingBox = False
                for box in self.boxes[self.idx]:
                    f = 0.4 #impose overlap by least this factor, horizontally and vertically
                    boxWidth = box[2]-box[0]
                    boxHeight = box[3]-box[1]
                    if (not box[2] < topLeftX     + boxWidth*f  and
                        not box[0] > bottomRightX - boxWidth*f  and
                        not box[3] < topLeftY     + boxHeight*f and
                        not box[1] > bottomRightY - boxHeight*f):
                        hasOverlappingBox = True
                        break
                self.outputs[fileIdx] += [[1,0] if hasOverlappingBox else [0,1]] * len(cellImages)
        if len(self.inputs[fileIdx]) == 0:
            raise Exception("No unfiltered cells for \"" + self.filenames[fileIdx] + "\"")
    #returns a tuple containing a numpy array of "size" inputs, and a numpy array of "size" outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            if self.valuesGenerated == self.VALUES_PER_IMAGE:
                self.idx = (self.idx + 1) % len(self.inputs)
                if LOAD_IMAGES_ON_DEMAND and self.inputs[self.idx] == None:
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
