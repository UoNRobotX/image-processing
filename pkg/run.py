import os, time, re, math
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import tensorflow as tf

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .network import createCoarseNetwork, createDetailedNetwork

def run(dataFile, filterFile, useCoarseOnly, reinitialise, outFile, threshold):
    #check if outputting text
    textOutput = re.search(r"\.txt$", outFile) != None
    if textOutput:
        if not useCoarseOnly:
            raise Exception("Text output not implemented for detailed network")
        else: #create and truncate file
            with open(outFile, "w"):
                pass
    #get input and output filenames
    if os.path.isfile(dataFile):
        filenames = [dataFile]
        outputFilenames = [outFile or "out.jpg"]
    elif os.path.isdir(dataFile):
        filenames = [
            dataFile + "/" + name for
            name in os.listdir(dataFile) if
            os.path.isfile(dataFile + "/" + name) and re.search(r"\.jpg$", name)
        ]
        filenames.sort()
        if textOutput:
            outputFilenames = [outFile] * len(filenames)
        else:
            outputDir = outFile or dataFile
            if not os.path.exists(outputDir):
                os.mkdir(outputDir)
            elif not os.path.isdir(outputDir):
                raise Exception("Invalid output directory")
            outputFilenames = [outputDir + "/" + os.path.basename(name) for name in filenames]
    else:
        raise Exception("Invalid input file")
    #initialise
    graph = tf.Graph()
    cellFilter = getCellFilter(filterFile)
    coarseNet = createCoarseNetwork(graph, threshold)
    detailedNet = None if useCoarseOnly else createDetailedNetwork(graph)
    with tf.Session(graph=graph) as sess:
        #reinitialise or load values
        sess.run(tf.initialize_all_variables())
        if useCoarseOnly:
            if not reinitialise and os.path.exists(COARSE_SAVE_FILE):
                tf.train.Saver(
                    tf.get_collection(tf.GraphKeys.VARIABLES, scope="coarse_net")
                ).restore(sess, COARSE_SAVE_FILE)
        else:
            if os.path.exists(COARSE_SAVE_FILE):
                tf.train.Saver(
                    tf.get_collection(tf.GraphKeys.VARIABLES, scope="coarse_net")
                ).restore(sess, COARSE_SAVE_FILE)
            if not reinitialise and os.path.exists(DETAILED_SAVE_FILE):
                tf.train.Saver(
                    tf.get_collection(tf.GraphKeys.VARIABLES, scope="detailed_net")
                ).restore(sess, DETAILED_SAVE_FILE)
        #process images
        for fileIdx in range(len(filenames)):
            filename = filenames[fileIdx]
            if useCoarseOnly:
                result = runCoarse(filename, cellFilter, coarseNet)
                writeCoarseResult(result, filename, outputFilenames[fileIdx], textOutput, threshold)
            else:
                result = runDetailed(filename, cellFilter, coarseNet, detailedNet, threshold)
                writeDetailedResult(result, filename, outputFilenames[fileIdx], textOutput)

def runCoarse(filename, cellFilter, coarseNet):
    startTime = time.time()
    #allocate results
    result = [
        [None for j in range(IMG_WIDTH//CELL_WIDTH)]
        for i in range(IMG_HEIGHT//CELL_HEIGHT)
    ] #elements will be None or a network output
    #obtain PIL image
    image = Image.open(filename)
    #get cells
    cellImgs = []
    cellIndices = []
    for row in range(IMG_HEIGHT//CELL_HEIGHT):
        for col in range(IMG_WIDTH//CELL_WIDTH):
            #filter with static filter
            if cellFilter[row][col] == 0:
                #get cell image
                cellImg = image.crop(
                    (col*CELL_WIDTH, row*CELL_HEIGHT, (col+1)*CELL_WIDTH, (row+1)*CELL_HEIGHT)
                )
                cellImg = cellImg.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.LANCZOS)
                cellImgs.append(cellImg)
                cellIndices.append([row, col])
    #obtain numpy arrays
    inputs = [
        np.array(list(cellImg.getdata())).astype(np.float32).reshape(
            (INPUT_HEIGHT, INPUT_WIDTH, IMG_CHANNELS)
        ) for cellImg in cellImgs
    ]
    preProcessingTime = time.time() - startTime
    #use network
    outputs = coarseNet.y.eval(feed_dict={coarseNet.x: inputs, coarseNet.p_dropout: 1.0})
    for i in range(len(outputs)):
        result[cellIndices[i][0]][cellIndices[i][1]] = outputs[i]
    processingTime = time.time() - startTime - preProcessingTime
    #print info
    print("Processed %s, pre-processing time %.2f secs, processing time %.2f secs" % \
        (filename, preProcessingTime, processingTime))
    #return results
    return result

def runDetailed(filename, cellFilter, coarseNet, detailedNet, threshold):
    startTime = time.time()
    #obtain PIL image
    image = Image.open(filename)
    #get cell data
    winPositions = GET_WINDOWS()
    winResults = [None for pos in winPositions]
        #each element will contain -2 (filtered), -1 (water), None (no object), 0+ (object)
    winImgs = []
    winImgIndices = []
    for winIdx in range(len(winPositions)):
        #get cell position
        pos = winPositions[winIdx]
        topLeftX     = pos[0]
        topLeftY     = pos[1]
        bottomRightX = pos[2]
        bottomRightY = pos[3]
        #use filter
        if isFiltered(topLeftX, topLeftY, bottomRightX, bottomRightY, cellFilter):
            winResults[winIdx] = -2
            continue
        #get window image
        winImg = image.crop((topLeftX, topLeftY, bottomRightX, bottomRightY))
        winImg = winImg.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.LANCZOS)
        winImgs.append(winImg)
        winImgIndices.append(winIdx)
    #obtain numpy arrays
    inputs = [
        np.array(winImg).astype(np.float32) #3D array, list of rows of lists of pixel values
        for winImg in winImgs
    ]
    if True: #use coarse network
        outputs = coarseNet.y.eval(feed_dict={coarseNet.x: inputs, coarseNet.p_dropout: 1.0})
        #remove filtered cells
        unfiltered = []
        for i in range(len(outputs)):
            if outputs[i][0] > threshold:
                winResults[winImgIndices[i]] = -1
            else:
                unfiltered.append(i)
        inputs = [inputs[i] for i in unfiltered]
        winImgIndices = [winImgIndices[i] for i in unfiltered]
    preProcessingTime = time.time() - startTime
    #use detailed network
    outputs = detailedNet.y.eval(feed_dict={detailedNet.x: inputs, detailedNet.p_dropout: 1.0})
    for i in range(len(outputs)):
        output = list(outputs[i])
        maxIdx = output.index(max(output))
        if maxIdx < NUM_BOX_TYPES:
            winResults[winImgIndices[i]] = maxIdx
    #get box lists
    filtered = [
        winPositions[i] for i in range(len(winPositions))
        if winResults[i] is not None and winResults[i] == -2
    ]
    coarseFiltered = [
        winPositions[i] for i in range(len(winPositions))
        if winResults[i] is not None and winResults[i] == -1
    ]
    predictions = [
        winPositions[i] + [winResults[i]] for i in range(len(winPositions))
        if winResults[i] is not None and winResults[i] >= 0
    ]
    #remove some overlapping boxes
    maxOverlapFactor = 0.5
    predictions.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True) #sort by area
    removed = [False for box in predictions]
    for i in range(len(predictions)):
        if removed[i]:
            continue
        for j in range(i+1, len(predictions)):
            if not removed[j]:
                b1 = predictions[i]
                b2 = predictions[j]
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])
                overlap = max(0, x2-x1) * max(0,y2-y1)
                overlapFactor = overlap / (b2[2] - b2[0]) * (b2[3] - b2[1])
                if overlapFactor > maxOverlapFactor:
                    removed[j] = True
    predictions = [predictions[i] for i in range(len(predictions)) if not removed[i]]
    #print info
    processingTime = time.time() - startTime - preProcessingTime
    print("Processed %s, pre-processing time %.2f secs, processing time %.2f secs" % \
        (filename, preProcessingTime, processingTime))
    #return results
    return filtered, coarseFiltered, predictions

def writeCoarseResult(result, filename, outputFilename, textOutput, threshold):
    FILTER_COLOR = (128, 0, 128, 128)
    POS_COLOR = (0, 255, 0, 96)
    NEG_COLOR = (255, 0, 0, 96)
    if textOutput:
        #write result to training/testing data file (used to bootstrap training)
        with open(outputFilename, "a") as file:
            file.write(filename + "\n")
            for row in result:
                file.write(" ")
                for cell in row:
                    if cell is not None and cell[0] > threshold:
                        file.write("1")
                    else:
                        file.write("0")
                file.write("\n")
    else:
        image = Image.open(filename)
        draw = ImageDraw.Draw(image, "RGBA")
        for i in range(IMG_HEIGHT//CELL_HEIGHT):
            for j in range(IMG_WIDTH//CELL_WIDTH):
                rect = [CELL_WIDTH*j, CELL_HEIGHT*i, CELL_WIDTH*(j+1), CELL_HEIGHT*(i+1)]
                #draw a grid cell outline
                draw.rectangle(rect, outline=(0,0,0,255))
                #draw a rectangle describing a cell's result
                output = result[i][j]
                if output is None:
                    draw.rectangle(rect, fill=FILTER_COLOR)
                else:
                    if True: #draw one rectangle
                        rect[1] += int(CELL_HEIGHT*(1-output[0]))
                        draw.rectangle(rect, fill=POS_COLOR)
                    else: #draw two rectangles describing both outputs
                        rect1 = rect.copy()
                        rect1[1] += int(CELL_HEIGHT*(1-output[0]))
                        rect1[2] -= CELL_WIDTH//2
                        rect2 = rect.copy()
                        rect2[0] += CELL_WIDTH//2
                        rect2[3] -= int(CELL_HEIGHT*(1-output[1]))
                        draw.rectangle(rect1, fill=POS_COLOR)
                        draw.rectangle(rect2, fill=NEG_COLOR)
        #save the image
        image.save(outputFilename)

def writeDetailedResult(result, filename, outputFilename, textOutput):
    FILTER_COLOR = (128, 0, 128, 32)
    COARSE_COLOR = (192, 160, 0, 32)
    if textOutput:
        raise Exception("Not implemented")
    else:
        image = Image.open(filename)
        draw = ImageDraw.Draw(image, "RGBA")
        filtered, coarseFiltered, predictions = result
        for box in filtered:
            draw.rectangle(box, fill=FILTER_COLOR)
        for box in coarseFiltered:
            draw.rectangle(box, fill=COARSE_COLOR)
        for box in predictions:
            col = tuple(list(ImageColor.getrgb(BOX_COLORS[box[4]])) + [64])
            draw.rectangle(box[0:4], outline="black", fill=col)
        #save the image
        image.save(outputFilename)

def isFiltered(topLeftX, topLeftY, bottomRightX, bottomRightY, cellFilter):
    for i in range(topLeftY // CELL_HEIGHT, bottomRightY // CELL_HEIGHT):
        for j in range(topLeftX // CELL_WIDTH, bottomRightX // CELL_WIDTH):
            if cellFilter[i][j]:
                return True
    return False

