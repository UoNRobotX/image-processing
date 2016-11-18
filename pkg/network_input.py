import math, random, re
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .constants import *

class CoarseBatchProducer:
    """ Produces input values for the coarse network """
    #constructor
    def __init__(self, dataFile, cellFilter, outFile=None):
        self.inputs  = None
        self.outputs = None
        #read "dataFile"
        if re.search(r"\.npz$", dataFile):
            data = np.load(dataFile)
            self.inputs = data["arr_0"]
            self.outputs = data["arr_1"]
        else:
            filenames = []
            waterCells = []
            with open(dataFile) as file:
                filename = None
                for line in file:
                    if line[0] != " ":
                        filename = line.strip()
                        filenames.append(filename)
                        waterCells.append([])
                    elif filename == None:
                        raise Exception("Invalid data file")
                    else:
                        waterCells[-1].append([int(c) for c in line.strip()])
            if len(filenames) == 0:
                raise Exception("No filenames")
            #allocate inputs and outputs
            self.inputs = [None for name in filenames]
            self.outputs = [None for name in filenames]
            #load images
            self.loadImages(filenames, cellFilter, waterCells)
        #save data if requested
        if outFile != None:
            np.savez_compressed(outFile, self.inputs, self.outputs)
    #load next image
    def loadImages(self, filenames, cellFilter, waterCells):
        for fileIdx in range(len(filenames)):
            #obtain PIL image
            image = Image.open(filenames[fileIdx])
            #allocate inputs and outputs
            self.inputs[fileIdx] = []
            self.outputs[fileIdx] = []
            for row in range(len(waterCells[fileIdx])):
                for col in range(len(waterCells[fileIdx][row])):
                    #use static filter
                    if cellFilter[row][col] == 1:
                        continue
                    #determine whether the input should have a positive prediction
                    containsWater = waterCells[fileIdx][row][col] == 1
                    ##randomly skip
                    #if not containsWater and random.random() > 0.25:
                    #    continue
                    #get cell image
                    cellImg = image.crop(
                        (col*CELL_WIDTH, row*CELL_HEIGHT, (col+1)*CELL_WIDTH, (row+1)*CELL_HEIGHT)
                    )
                    cellImg = cellImg.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.LANCZOS)
                    #preprocess image
                    cellImgs = [cellImg]
                    if False: #maximise image contrast
                        cellImgs = [ImageOps.autocontrast(img) for img in cellImgs]
                    if False: #equalize image histogram
                        cellImgs = [ImageOps.equalize(img) for img in cellImgs]
                    if False: #blur image
                        cellImgs = [img.filter(ImageFilter.GaussianBlur(radius=2)) for img in cellImgs]
                    if False: #sharpen
                        cellImgs = [
                            img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                            for img in cellImgs
                        ]
                    if False: #use kernel
                        cellImgs = [
                            img.filter(ImageFilter.Kernel((3,3), (0, 0, 0, 0, 1, 0, 0, 0, 0)))
                            for img in cellImgs
                    ]
                    if False: #other
                        cellImgs = [img.filter(ImageFilter.FIND_EDGES) for img in cellImgs]
                    if False and not containsWater: #add rotated images
                        cellImgs += [img.rotate(180) for img in cellImgs]
                        cellImgs += [img.rotate(90) for img in cellImgs]
                    if True and not containsWater: #add flipped images
                        cellImgs += [img.transpose(Image.FLIP_LEFT_RIGHT) for img in cellImgs]
                    if False: #add sheared images
                        shearFactor = random.random()*0.8 - 0.4
                        cellImgs += [
                            img.transform(
                                (img.size[0], img.size[1]),
                                Image.AFFINE,
                                data=(
                                    (1-shearFactor, shearFactor, 0, 0, 1, 0) if shearFactor>0 else
                                    (1+shearFactor, shearFactor, -shearFactor*img.size[0], 0, 1, 0)
                                ),
                                resample=Image.BICUBIC)
                            for img in cellImgs
                        ]
                    #get inputs
                    self.inputs[fileIdx] += [
                        np.array(list(img.getdata())).astype(np.float32).reshape(
                            (INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS)
                        )
                        for img in cellImgs
                    ]
                    #get outputs
                    self.outputs[fileIdx] += [
                        np.array([1, 0]).astype(np.float32) if containsWater else
                        np.array([0, 1]).astype(np.float32)
                    ] * len(cellImgs)
            if len(self.inputs[fileIdx]) == 0:
                raise Exception("No unfiltered cells for \"" + filenames[fileIdx] + "\"")
    #returns a tuple containing a numpy array of "size" inputs, and a numpy array of "size" outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            #randomly select an input and output
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
        return np.array(inputs), np.array(outputs)
    #returns the data set size
    def getDatasetSize(self):
        numInputs = 0
        for i in range(len(self.inputs)):
            numInputs += len(self.inputs[i])
        return numInputs

class DetailedBatchProducer:
    """Produces input values for the detailed network"""
    #constructor
    def __init__(self, dataFile, cellFilter, outFile=None):
        self.inputs  = None
        self.outputs = None
        #read "dataFile"
        if re.search(r"\.npz$", dataFile):
            data = np.load(dataFile)
            self.inputs = data["arr_0"]
            self.outputs = data["arr_1"]
        else:
            filenames = []
            boxes = []
            with open(dataFile) as file:
                filename = None
                for line in file:
                    if line[0] != " ":
                        filename = line.strip()
                        filenames.append(filename)
                        boxes.append([])
                    elif filename == None:
                        raise Exception("Invalid data file")
                    else:
                        boxes[-1].append([int(c) for c in line.strip().split(",")])
            if len(filenames) == 0:
                raise Exception("No filenames")
            #allocate inputs and outputs
            self.inputs = [None for name in filenames]
            self.outputs = [None for name in filenames]
            #load images
            self.loadImages(filenames, cellFilter, boxes)
        #save data if requested
        if outFile != None:
            np.savez_compressed(outFile, self.inputs, self.outputs)
    #load next image
    def loadImages(self, filenames, cellFilter, boxes):
        for fileIdx in range(len(filenames)):
            #obtain PIL image
            image = Image.open(filenames[fileIdx])
            #allocate inputs and outputs
            self.inputs[fileIdx] = []
            self.outputs[fileIdx] = []
            #get window positions
            windowPositions = GET_WINDOWS()
            for pos in windowPositions:
                topLeftX = pos[0]
                topLeftY = pos[1]
                bottomRightX = pos[2]
                bottomRightY = pos[3]
                #use static filter
                if isFiltered(topLeftX, topLeftY, bottomRightX, bottomRightY, cellFilter):
                    continue
                #determine whether the input should have a positive prediction
                containsBuoy = False
                for box in boxes[fileIdx]:
                    if True: #only accept if a buoy is fully contained
                        if (box[0] >= topLeftX and
                            box[1] >= topLeftY and
                            box[2] <= bottomRightX and
                            box[3] <= bottomRightY):
                            #and is no less than a minimum overlap ratio
                            boxArea = (box[2]-box[0]) * (box[3]-box[1])
                            winArea = (bottomRightX-topLeftX) * (bottomRightY-topLeftY)
                            if boxArea / winArea >= 0.3:
                                containsBuoy = True
                                break
                    else: #impose overlap by least some factor, horizontally and vertically
                        f = 0.4
                        boxWidth = box[2]-box[0]
                        boxHeight = box[3]-box[1]
                        if (not box[2] < topLeftX     + boxWidth*f  and
                            not box[0] > bottomRightX - boxWidth*f  and
                            not box[3] < topLeftY     + boxHeight*f and
                            not box[1] > bottomRightY - boxHeight*f):
                            containsBuoy = True
                            break
                #randomly skip
                if not containsBuoy and random.random() > 0.3:
                    continue
                #get window image
                winImg = image.crop((topLeftX, topLeftY, bottomRightX, bottomRightY))
                winImg = winImg.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.LANCZOS)
                #preprocess image
                winImgs = [winImg]
                if False: #maximise image contrast
                    winImgs = [ImageOps.autocontrast(img) for img in winImgs]
                if True and containsBuoy: #add rotated images
                    winImgs += [img.rotate(180) for img in winImgs]
                    winImgs += [img.rotate(90) for img in winImgs]
                if True and containsBuoy: #add flipped images
                    winImgs += [img.transpose(Image.FLIP_LEFT_RIGHT) for img in winImgs]
                if True and containsBuoy: #blur image
                    blurRadii = [1.0]
                    blurredImages = []
                    for radius in blurRadii:
                        blurredImages += [
                            img.filter(ImageFilter.GaussianBlur(radius)) for img in winImgs
                        ]
                    winImgs += blurredImages
                if True and containsBuoy: #add sheared images
                    shearedImages = []
                    for maxShearFactor in [0.1, 0.2]:
                        shearFactor = random.random()*maxShearFactor*2 - maxShearFactor
                        shearedImages += [
                            img.transform(
                                (img.size[0], img.size[1]),
                                Image.AFFINE,
                                data=(
                                    (1-shearFactor, shearFactor, 0, 0, 1, 0) if shearFactor>0 else
                                    (1+shearFactor, shearFactor, -shearFactor*img.size[0], 0, 1, 0)
                                ),
                                resample=Image.BICUBIC)
                            for img in winImgs
                        ]
                    winImgs += shearedImages
                #get inputs
                self.inputs[fileIdx] += [
                    np.array(list(img.getdata())).astype(np.float32).reshape(
                        (INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS)
                    )
                    for img in winImgs
                ]
                #get outputs
                self.outputs[fileIdx] += [
                    np.array([1, 0]).astype(np.float32) if containsBuoy else
                    np.array([0, 1]).astype(np.float32)
                ] * len(winImgs)
            if len(self.inputs[fileIdx]) == 0:
                raise Exception("No inputs for \"" + filenames[fileIdx] + "\"")
    #returns a tuple containing a numpy array of "size" inputs, and a numpy array of "size" outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        c = 0
        while c < size:
            #randomly select an input and output
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
        return np.array(inputs), np.array(outputs)
    #returns the data set size
    def getDatasetSize(self):
        numInputs = 0
        for i in range(len(self.inputs)):
            numInputs += len(self.inputs[i])
        return numInputs

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
            [0 for col in IMG_WIDTH // CELL_WIDTH]
            for row in IMG_HEIGHT // CELL_HEIGHT
        ]
    return cellFilter

def isFiltered(topLeftX, topLeftY, bottomRightX, bottomRightY, cellFilter):
    for i in range(topLeftY // CELL_HEIGHT, bottomRightY // CELL_HEIGHT):
        for j in range(topLeftX // CELL_WIDTH, bottomRightX // CELL_WIDTH):
            if cellFilter[i][j]:
                return True
    return False
