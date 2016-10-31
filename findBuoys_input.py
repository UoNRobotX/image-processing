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
    #constructor
    def __init__(self, dataFile, cellFilter):
        self.filenames = [] #list of image files
        self.cells = []     #has the form [[[c1, c2, ...], ...], ...], specifying cells of images
        self.fileIdx = 0
        self.image = None
        self.valuesGenerated = 0
        self.unfilteredCells = None
        #read 'dataFile'
        cellsDict = dict()
        filename = None
        with open(dataFile) as file:
            for line in file:
                if line[0] != " ":
                    filename = line.strip()
                    self.filenames.append(filename)
                    cellsDict[filename] = []
                else:
                    cellsDict[filename].append([int(c) for c in line.strip()])
        random.shuffle(self.filenames)
        self.cells = [cellsDict[name] for name in self.filenames]
        if len(self.filenames) == 0:
            raise Exception("No filenames")
        #obtain PIL image
        self.image = Image.open(self.filenames[self.fileIdx])
        self.image = self.image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #obtain indices of non-filtered cells (used to randomly select a non-filtered cell)
        rowSize = IMG_SCALED_WIDTH//INPUT_WIDTH
        colSize = IMG_SCALED_HEIGHT//INPUT_HEIGHT
        self.unfilteredCells = []
        for row in range(len(cellFilter)):
            for col in range(len(cellFilter[row])):
                if cellFilter[row][col] == 0:
                    self.unfilteredCells.append(col+row*rowSize)
        if len(self.unfilteredCells) == 0:
            raise Exception("No unfiltered cells")
    #returns a tuple containing a numpy array of 'size' inputs, and a numpy array of 'size' outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            if self.valuesGenerated == self.VALUES_PER_IMAGE:
                #open next image file
                self.fileIdx += 1
                if self.fileIdx+1 > len(self.filenames):
                    self.fileIdx = 0
                self.image = Image.open(self.filenames[self.fileIdx])
                self.image = self.image.resize(
                    (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
                    resample=Image.LANCZOS
                )
                self.valuesGenerated = 0
            #randomly select a non-filtered grid cell
            idx = self.unfilteredCells[
                math.floor(random.random() * len(self.unfilteredCells))
            ]
            rowSize = IMG_SCALED_WIDTH // INPUT_WIDTH
            i = idx % rowSize
            j = idx // rowSize
            x = i*INPUT_WIDTH
            y = j*INPUT_HEIGHT
            ##bias samples towards positive examples
            #if self.cells[self.fileIdx][j][i] == 0 and random.random() < 0.5:
            #    continue
            #get an input
            cellImg = self.image.crop((x, y, x+INPUT_WIDTH, y+INPUT_HEIGHT))
            cellImg = cellImg.rotate(math.floor(random.random() * 4) * 90) #randomly rotate
            if random.random() > 0.5: #randomly flip
                cellImg = cellImg.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5: #randomly flip
                cellImg = cellImg.transpose(Image.FLIP_TOP_BOTTOM)
            #cellImg = ImageOps.autocontrast(cellImg)
            #cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
            data = np.array(list(cellImg.getdata())).astype(np.float32)
            data = data.reshape((INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS))
            inputs.append(data)
            #get an output
            outputs.append([1, 0] if self.cells[self.fileIdx][j][i] == 1 else [0, 1])
            #update
            self.valuesGenerated += 1
            c += 1
        return np.array(inputs), np.array(outputs).astype(np.float32)

#class for producing detailed network input values from a training/test data file
class BatchProducer:
    """Produces input values for the detailed network"""
    VALUES_PER_IMAGE = 100
    #constructor
    def __init__(self, dataFile, cellFilter, coarseX, coarseY):
        self.filenames = [] #list of image files
        self.boxes = []     #has the form [[x,y,x2,y2], ...], and specifies boxes for each image file
        self.fileIdx = 0
        self.image = None
        self.valuesGenerated = 0
        self.unfilteredCells = None
        self.coarseX = coarseX
        self.coarseY = coarseY #allows using the coarse network to filter cells
        #read 'dataFile'
        filenameSet = set()
        boxesDict = dict()
        filename = None
        with open(dataFile) as file:
            for line in file:
                if line[0] != " ":
                    filename = line.strip()
                    filenameSet.add(filename)
                    if not filename in boxesDict:
                        boxesDict[filename] = []
                else:
                    boxesDict[filename].append([int(c) for c in line.strip().split(",")])
        self.filenames = list(filenameSet)
        random.shuffle(self.filenames)
        self.boxes = [boxesDict[name] for name in self.filenames]
        if len(self.filenames) == 0:
            raise Exception("No filenames")
        #obtain PIL image
        self.image = Image.open(self.filenames[self.fileIdx])
        self.image = self.image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #obtain indices of non-filtered cells (used to randomly select a non-filtered cell)
        rowSize = IMG_SCALED_WIDTH//INPUT_WIDTH
        colSize = IMG_SCALED_HEIGHT//INPUT_HEIGHT
        self.unfilteredCells = []
        for row in range(len(cellFilter)):
            for col in range(len(cellFilter[row])):
                if cellFilter[row][col] == 0:
                    self.unfilteredCells.append(col+row*rowSize)
        if len(self.unfilteredCells) == 0:
            raise Exception("No unfiltered cells")
    #returns a tuple containing a numpy array of 'size' inputs, and a numpy array of 'size' outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        potentialInputs = []
        potentialOutputs = []
        while size > 0:
            for i in range(size):
                if self.valuesGenerated == self.VALUES_PER_IMAGE:
                    #open next image file
                    self.fileIdx += 1
                    if self.fileIdx+1 > len(self.filenames):
                        self.fileIdx = 0
                    self.image = Image.open(self.filenames[self.fileIdx])
                    self.image = self.image.resize(
                        (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
                        resample=Image.LANCZOS
                    )
                    self.valuesGenerated = 0
                #randomly select a non-filtered grid cell
                idx = self.unfilteredCells[
                    math.floor(random.random() * len(self.unfilteredCells))
                ]
                rowSize = IMG_SCALED_WIDTH // INPUT_WIDTH
                i = idx % rowSize
                j = idx // rowSize
                x = i*INPUT_WIDTH
                y = j*INPUT_HEIGHT
                #get an input
                cellImg = self.image.crop((x, y, x+INPUT_WIDTH, y+INPUT_HEIGHT))
                cellImg = cellImg.rotate(math.floor(random.random() * 4) * 90) #randomly rotate
                if random.random() > 0.5: #randomly flip
                    cellImg = cellImg.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() > 0.5: #randomly flip
                    cellImg = cellImg.transpose(Image.FLIP_TOP_BOTTOM)
                #cellImg = ImageOps.autocontrast(cellImg)
                #cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
                data = np.array(list(cellImg.getdata())).astype(np.float32)
                data = data.reshape((INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS))
                potentialInputs.append(data)
                #get an output
                topLeftX = x*IMG_DOWNSCALE + 15
                topLeftY = y*IMG_DOWNSCALE + 15
                bottomRightX = (x+INPUT_WIDTH-1)*IMG_DOWNSCALE - 15
                bottomRightY = (y+INPUT_HEIGHT-1)*IMG_DOWNSCALE - 15
                hasOverlappingBox = False
                for box in self.boxes[self.fileIdx]:
                    if (not box[2] < topLeftX and
                        not box[0] > bottomRightX and
                        not box[3] < topLeftY and
                        not box[1] > bottomRightY):
                        hasOverlappingBox = True
                        break
                potentialOutputs.append([1, 0] if hasOverlappingBox else [0, 1])
                #update
                self.valuesGenerated += 1
            #filter using coarse network
            out = self.coarseY.eval(feed_dict={self.coarseX: np.array(potentialInputs)})
            unfilteredIndices = [i for i in range(len(potentialInputs)) if out[i][0] < threshold]
            inputs  += [potentialInputs[i] for i in unfilteredIndices]
            outputs += [potentialOutputs[i] for i in unfilteredIndices]
            #update
            size -= len(unfilteredIndices)
            potentialInputs = []
            potentialOutputs = []
        return np.array(inputs), np.array(outputs).astype(np.float32)

