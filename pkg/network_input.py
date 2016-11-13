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
            [0 for col in IMG_WIDTH // CELL_WIDTH]
            for row in IMG_HEIGHT // CELL_HEIGHT
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
            cells = []
            with open(dataFile) as file:
                filename = None
                for line in file:
                    if line[0] != " ":
                        filename = line.strip()
                        filenames.append(filename)
                        cells.append([])
                    elif filename == None:
                        raise Exception("Invalid data file")
                    else:
                        cells[-1].append([int(c) for c in line.strip()])
            if len(filenames) == 0:
                raise Exception("No filenames")
            #allocate inputs and outputs
            self.inputs = [None for name in filenames]
            self.outputs = [None for name in filenames]
            #load images
            self.loadImages(filenames, cellFilter, cells)
        #save data if requested
        if outFile != None:
            np.savez_compressed(outFile, self.inputs, self.outputs)
    #load next image
    def loadImages(self, filenames, cellFilter, cells):
        for fileIdx in range(len(filenames)):
            #obtain PIL image
            image = Image.open(filenames[fileIdx])
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
                        (col*CELL_WIDTH, row*CELL_HEIGHT, (col+1)*CELL_WIDTH, (row+1)*CELL_HEIGHT)
                    )
                    #downscale
                    cellImg = cellImg.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.LANCZOS)
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
                    self.outputs[fileIdx] += [
                        np.array([1, 0]).astype(np.float32) if containsWater else
                        np.array([0, 1]).astype(np.float32)
                    ] * len(cellImages)
            if len(self.inputs[fileIdx]) == 0:
                raise Exception("No unfiltered cells for \"" + filenames[fileIdx] + "\"")
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
        return np.array(inputs), np.array(outputs)

class DetailedBatchProducer:
    """Produces input values for the detailed network"""
    #constructor
    def __init__(self, dataFile, cellFilter, outFile):
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
            #get inputs and outputs
            self.inputs[fileIdx] = []
            self.outputs[fileIdx] = []
            #get cell positions
            cellPositions = GET_VAR_CELLS()
            for pos in cellPositions:
                #get cell position
                topLeftX = pos[0]
                topLeftY = pos[1]
                bottomRightX = pos[2]
                bottomRightY = pos[3]
                #use static filter
                hasOverlappingFilteredCell = False
                for i in range(topLeftY // CELL_HEIGHT, bottomRightY // CELL_HEIGHT):
                    for j in range(topLeftX // CELL_WIDTH, bottomRightX // CELL_WIDTH):
                        if cellFilter[i][j] == 1:
                            hasOverlappingFilteredCell = True
                            break
                    if hasOverlappingFilteredCell:
                        break
                #get cell image
                cellImg = image.crop((topLeftX, topLeftY, bottomRightX, bottomRightY))
                cellImg = cellImg.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.LANCZOS)
                #determine whether the input should have a positive prediction
                containsBuoy = False
                for box in boxes[fileIdx]:
                    if True: #impose overlap by least some factor, horizontally and vertically
                        f = 0.4
                        boxWidth = box[2]-box[0]
                        boxHeight = box[3]-box[1]
                        if (not box[2] < topLeftX     + boxWidth*f  and
                            not box[0] > bottomRightX - boxWidth*f  and
                            not box[3] < topLeftY     + boxHeight*f and
                            not box[1] > bottomRightY - boxHeight*f):
                            containsBuoy = True
                            break
                    else: #only accept if a buoy is fully contained
                        if (box[0] >= topLeftX and
                            box[1] >= topLeftY and
                            box[2] <= bottomRightX and
                            box[3] <= bottomRightY):
                            containsBuoy = True
                            break
                #preprocess image
                if False: #maximise image contrast
                    cellImg = ImageOps.autocontrast(cellImg)
                if False: #blur image
                    cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
                cellImages = [cellImg]
                if False and containsBuoy: #add rotated images
                    cellImages += [cellImg.rotate(180) for img in cellImages]
                    cellImages += [cellImg.rotate(90) for img in cellImages]
                if False and containsBuoy: #add flipped images
                    cellImages += [cellImg.transpose(Image.FLIP_LEFT_RIGHT) for img in cellImages]
                if False and containsBuoy: #add sheared images
                    for maxShearFactor in [0.1, 0.2]:
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
                self.outputs[fileIdx] += [
                    np.array([1, 0]).astype(np.float32) if containsBuoy else
                    np.array([0, 1]).astype(np.float32)
                ] * len(cellImages)
            if len(self.inputs[fileIdx]) == 0:
                raise Exception("No unfiltered cells for \"" + filenames[fileIdx] + "\"")
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
        return np.array(inputs), np.array(outputs)
