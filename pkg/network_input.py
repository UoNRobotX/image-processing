import math, random, re
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
    #constructor
    def __init__(self, dataFile, cellFilter, outFile):
        self.inputs          = None
        self.outputs         = None
        #read "dataFile"
        if re.search(r"\.npz$", dataFile):
            data = np.load(dataFile)
            self.inputs = data["arr_0"]
            self.outputs = data["arr_1"]
        else:
            filenames = []
            cellsDict = dict()
            with open(dataFile) as file:
                filename = None
                for line in file:
                    if line[0] != " ":
                        filename = line.strip()
                        filenames.append(filename)
                        cellsDict[filename] = []
                    elif filename == None:
                        raise Exception("Invalid data file")
                    else:
                        cellsDict[filename].append([int(c) for c in line.strip()])
            if len(filenames) == 0:
                raise Exception("No filenames")
            cells = [cellsDict[name] for name in filenames]
            #allocate inputs and outputs
            self.inputs = [None for name in filenames]
            self.outputs = [None for name in filenames]
            #load images
            for i in range(len(filenames)):
                self.loadImage(i, filenames[i], cellFilter, cells)
        #save data if requested
        if outFile != None:
            np.savez_compressed(outFile, self.inputs, self.outputs)
    #load next image
    def loadImage(self, fileIdx, filename, cellFilter, cells):
        #obtain PIL image
        image = Image.open(filename)
        image = image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #get inputs and outputs
        self.inputs[fileIdx] = []
        self.outputs[fileIdx] = []
        for row in range(len(cells[fileIdx])):
            for col in range(len(cells[fileIdx][row])):
                #use static filter
                if cellFilter[row][col] == 1:
                    continue
                #get cell image
                cellImg = image.crop(
                    (col*INPUT_WIDTH, row*INPUT_HEIGHT, (col+1)*INPUT_WIDTH, (row+1)*INPUT_HEIGHT)
                )
                #determine whether the input should have a positive prediction
                containsWater = cells[fileIdx][row][col] == 1
                #preprocess image
                if False: #maximise image contrast
                    cellImg = ImageOps.autocontrast(cellImg)
                if False: #equalize image histogram
                    cellImg = ImageOps.equalize(cellImg)
                if False: #blur image
                    cellImg = cellImg.filter(ImageFilter.GaussianBlur(radius=2))
                if False: #sharpen
                    cellImg = cellImg.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                if False: #min/max/median/mode filter
                    cellImg.filter(ImageFilter.MinFilter(size=3))
                    #cellImg.filter(ImageFilter.MaxFilter(size=3))
                    #cellImg.filter(ImageFilter.MedianFilter(size=3))
                    #cellImg.filter(ImageFilter.ModeFilter(size=3))
                if False: #use kernel
                    cellImg = cellImg.filter(ImageFilter.Kernel((3,3), (0, 0, 0, 0, 1, 0, 0, 0, 0)))
                if False: #other
                    cellImg = cellImg.filter(ImageFilter.FIND_EDGES)
                cellImages = [cellImg]
                if False: #add rotated images
                    cellImages += [cellImg.rotate(180) for img in cellImages]
                    cellImages += [cellImg.rotate(90) for img in cellImages]
                if False and containsWater: #add flip images
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
                self.outputs[fileIdx] += [[1,0] if containsWater else [0,1]] * len(cellImages)
        if len(self.inputs[fileIdx]) == 0:
            raise Exception("No unfiltered cells for \"" + filename + "\"")
    #returns a tuple containing a numpy array of "size" inputs, and a numpy array of "size" outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            #randomly select a non-filtered grid cell
            fileIdx = math.floor(random.random() * len(self.inputs))
            idx = math.floor(random.random() * len(self.inputs[fileIdx]))
            ##bias samples towards positive examples
            #if self.outputs[fileIdx][idx][0] == 0 and random.random() < 0.5:
            #    continue
            #add input and output to batch
            inputs.append(self.inputs[fileIdx][idx])
            outputs.append(self.outputs[fileIdx][idx])
            #update
            c += 1
        return np.array(inputs), np.array(outputs).astype(np.float32)

class DetailedBatchProducer:
    """Produces input values for the detailed network"""
    #constructor
    def __init__(self, dataFile, cellFilter, outFile):
        self.inputs          = None
        self.outputs         = None
        #read "dataFile"
        if re.search(r"\.npz$", dataFile):
            data = np.load(dataFile)
            self.inputs = data["arr_0"]
            self.outputs = data["arr_1"]
        else:
            filenames = []
            boxesDict = dict()
            with open(dataFile) as file:
                filename = None
                for line in file:
                    if line[0] != " ":
                        filename = line.strip()
                        filenames.append(filename)
                        boxesDict[filename] = []
                    elif filename == None:
                        raise Exception("Invalid data file")
                    else:
                        boxesDict[filename].append([int(c) for c in line.strip().split(",")])
            if len(filenames) == 0:
                raise Exception("No filenames")
            boxes = [boxesDict[name] for name in filenames]
            #allocate inputs and outputs
            self.inputs = [None for name in filenames]
            self.outputs = [None for name in filenames]
            #load images
            for i in range(len(filenames)):
                self.loadImage(i, filenames[i], cellFilter, boxes)
        #save data if requested
        if outFile != None:
            np.savez_compressed(outFile, self.inputs, self.outputs)
    #load next image
    def loadImage(self, fileIdx, filename, cellFilter, boxes):
        #obtain PIL image
        image = Image.open(filename)
        image = image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #get inputs and outputs
        self.inputs[fileIdx] = []
        self.outputs[fileIdx] = []
        stride = (INPUT_WIDTH//1, INPUT_HEIGHT//1)
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
                    if cellFilter[row][col] == 1:
                        hasOverlappingFilteredCell = True
                        break
                if hasOverlappingFilteredCell:
                    continue
                #get cell image
                cellImg = image.crop(
                    (x, y, x+INPUT_WIDTH, y+INPUT_HEIGHT)
                )
                #TODO: filter with coarse network?
                #determine whether the input should have a positive prediction
                topLeftX = x*IMG_DOWNSCALE
                topLeftY = y*IMG_DOWNSCALE
                bottomRightX = (x+INPUT_WIDTH-1)*IMG_DOWNSCALE
                bottomRightY = (y+INPUT_HEIGHT-1)*IMG_DOWNSCALE
                containsBuoy = False
                for box in boxes[fileIdx]:
                    f = 0.4 #impose overlap by least this factor, horizontally and vertically
                    boxWidth = box[2]-box[0]
                    boxHeight = box[3]-box[1]
                    if (not box[2] < topLeftX     + boxWidth*f  and
                        not box[0] > bottomRightX - boxWidth*f  and
                        not box[3] < topLeftY     + boxHeight*f and
                        not box[1] > bottomRightY - boxHeight*f):
                        containsBuoy = True
                        break
                #preprocess image
                if False: #maximise image contrast
                    cellImg = ImageOps.autocontrast(cellImg)
                if False: #blur image
                    cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
                cellImages = [cellImg]
                if False: #add rotated images
                    cellImages += [cellImg.rotate(180) for img in cellImages]
                    cellImages += [cellImg.rotate(90) for img in cellImages]
                if False and containsBuoy: #add flipped images
                    cellImages += [cellImg.transpose(Image.FLIP_LEFT_RIGHT) for img in cellImages]
                if False and containsBuoy: #add sheared images
                    for maxShearFactor in [0.2, 0.2]:
                        shearFactor = random.random()*maxShearFactor*2 - maxShearFactor
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
                self.outputs[fileIdx] += [[1,0] if containsBuoy else [0,1]] * len(cellImages)
        if len(self.inputs[fileIdx]) == 0:
            raise Exception("No unfiltered cells for \"" + filename + "\"")
    #returns a tuple containing a numpy array of "size" inputs, and a numpy array of "size" outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            #randomly select a non-filtered grid cell
            fileIdx = math.floor(random.random() * len(self.inputs))
            idx = math.floor(random.random() * len(self.inputs[fileIdx]))
            ##bias samples towards positive examples
            #if self.outputs[fileIdx][idx][0] == 0 and random.random() < 0.5:
            #    continue
            #add input and output to batch
            inputs.append(self.inputs[fileIdx][idx])
            outputs.append(self.outputs[fileIdx][idx])
            #update
            c += 1
        return np.array(inputs), np.array(outputs).astype(np.float32)
