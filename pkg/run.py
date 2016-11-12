import os, time, re
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .network import createCoarseNetwork, createDetailedNetwork

def run(dataFile, filterFile, useCoarseOnly, reinitialise, outFile, threshold):
    #check if outputting text
    textOutput = re.search(r"\.txt$", outFile) != None
    if textOutput and not useCoarseOnly:
        raise Exception("Text output not implemented for detailed network")
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
            outputFilenames = [outFile]
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
    cellFilter = getCellFilter(filterFile)
    coarseNet = createCoarseNetwork(threshold)
    detailedNet = None if useCoarseOnly else createDetailedNetwork()
    #process images
    results = []
    for fileIdx in range(len(filenames)):
        filename = filenames[fileIdx]
        if useCoarseOnly:
            result = runCoarse(filename, cellFilter, coarseNet, reinitialise)
        else:
            result = runDetailed(filename, cellFilter, coarseNet, detailedNet, reinitialise, threshold)
        results.append(result)
    #output results
    if useCoarseOnly:
        writeCoarseResults(results, filenames, outputFilenames, textOutput)
    else:
        writeDetailedResults(results, filenames, outputFilenames, textOutput)

def runCoarse(filename, cellFilter, coarseNet, reinitialise):
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
    outputs = runNetwork(coarseNet, inputs, reinitialise, COARSE_SAVE_FILE)
    for i in range(len(outputs)):
        result[cellIndices[i][0]][cellIndices[i][1]] = outputs[i]
    processingTime = time.time() - startTime - preProcessingTime
    #print info
    print("Processed %s, pre-processing time %.2f secs, processing time %.2f secs" % \
        (filename, preProcessingTime, processingTime))
    #return results
    return result

def runDetailed(filename, cellFilter, coarseNet, detailedNet, reinitialise, threshold):
    startTime = time.time()
    #obtain PIL image
    image = Image.open(filename)
    #get cell data
    cellPositions = GET_VAR_CELLS()
    cellResults = [None for pos in cellPositions]
        #contains -2 (static filtered), -1 (coarse filtered), or an output
    cellImgs = []
    cellImgIndices = []
    for cellIdx in range(len(cellPositions)):
        #get cell position
        topLeftX = cellPositions[cellIdx][0]
        topLeftY = cellPositions[cellIdx][1]
        bottomRightX = cellPositions[cellIdx][2]
        bottomRightY = cellPositions[cellIdx][3]
        #use static filter
        hasOverlappingFilteredCell = False
        for i in range(topLeftY // CELL_HEIGHT, bottomRightY // CELL_HEIGHT):
            for j in range(topLeftX // CELL_WIDTH, bottomRightX // CELL_WIDTH):
                if cellFilter[i][j] == 1:
                    hasOverlappingFilteredCell = True
                    break
            if hasOverlappingFilteredCell:
                break
        if hasOverlappingFilteredCell:
            cellResults[cellIdx] = -2
            continue
        #get cell image
        cellImg = image.crop((topLeftX, topLeftY, bottomRightX, bottomRightY))
        cellImg = cellImg.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.LANCZOS)
        cellImgs.append(cellImg)
        cellImgIndices.append(cellIdx)
    #obtain numpy arrays
    inputs = [
        np.array(list(cellImg.getdata())).astype(np.float32).reshape(
            (INPUT_HEIGHT, INPUT_WIDTH, IMG_CHANNELS)
        ) for cellImg in cellImgs
    ]
    if False: #use coarse network
        outputs = runNetwork(coarseNet, inputs, False, COARSE_SAVE_FILE)
        #remove filtered cells
        unfiltered = []
        for i in range(len(outputs)):
            if outputs[i][0] > threshold:
                cellResults[cellImgIndices[i]] = -1
            else:
                unfiltered.append(i)
        inputs = [inputs[i] for i in unfiltered]
        cellImgIndices = [cellImgIndices[i] for i in unfiltered]
    preProcessingTime = time.time() - startTime
    #use detailed network
    outputs = runNetwork(detailedNet, inputs, reinitialise, DETAILED_SAVE_FILE)
    for i in range(len(outputs)):
        cellResults[cellImgIndices[i]] = outputs[i]
    processingTime = time.time() - startTime - preProcessingTime
    #print info
    print("Processed %s, pre-processing time %.2f secs, processing time %.2f secs" % \
        (filename, preProcessingTime, processingTime))
    #return results
    return (cellPositions, cellResults)

def runNetwork(net, inputs, reinitialise, saveFile):
    with tf.Session(graph=net.graph) as sess:
        #reinitialise or load values
        if reinitialise or not os.path.exists(saveFile):
            sess.run(tf.initialize_all_variables())
        else:
            tf.train.Saver(tf.all_variables()).restore(sess, saveFile)
        #run
        outputs = net.y.eval(feed_dict={
            net.x: inputs,
            net.p_dropout: 1.0
        })
        return outputs

def writeCoarseResults(results, filenames, outputFilenames, textOutput):
    FILTER_COLOR = (128, 0, 128, 128)
    POS_COLOR = (0, 255, 0, 96)
    NEG_COLOR = (255, 0, 0, 96)
    if textOutput:
        #write results to training/testing data file (used to bootstrap training)
        with open(outputFilenames[0], "w") as file:
            for fileIdx in len(filenames):
                file.write(filenames[fileIdx] + "\n")
                for row in results[fileIdx]:
                    file.write(" ")
                    for cell in row:
                        if cell != None and cell[0] > threshold:
                            file.write("1")
                        else:
                            file.write("0")
                    file.write("\n")
    else:
        for fileIdx in range(len(filenames)):
            image = Image.open(filenames[fileIdx])
            draw = ImageDraw.Draw(image, "RGBA")
            result = results[fileIdx]
            for i in range(IMG_HEIGHT//CELL_HEIGHT):
                for j in range(IMG_WIDTH//CELL_WIDTH):
                    rect = [CELL_WIDTH*j, CELL_HEIGHT*i, CELL_WIDTH*(j+1), CELL_HEIGHT*(i+1)]
                    #draw a grid cell outline
                    draw.rectangle(rect, outline=(0,0,0,255))
                    #draw a rectangle describing the results
                    output = result[i][j]
                    if output is None:
                        draw.rectangle(rect, fill=FILTER_COLOR)
                    else:
                        rect1 = rect.copy()
                        rect1[1] += int(CELL_HEIGHT*(1-output[0]))
                        rect1[2] -= CELL_WIDTH//2
                        rect2 = rect.copy()
                        rect2[0] += CELL_WIDTH//2
                        rect2[3] -= int(CELL_HEIGHT*(1-output[1]))
                        draw.rectangle(rect1, fill=POS_COLOR)
                        draw.rectangle(rect2, fill=NEG_COLOR)
            #save the image
            image.save(outputFilenames[fileIdx])
            print("Wrote %s" % outputFilenames[fileIdx])

def writeDetailedResults(results, filenames, outputFilenames, textOutput):
    FILTER_COLOR = (128, 0, 128, 128)
    COARSE_COLOR = (192, 160, 0, 128)
    POS_COLOR = (0, 255, 0, 96)
    if textOutput:
        raise Exception("Not implemented")
    else:
        for fileIdx in range(len(filenames)):
            image = Image.open(filenames[fileIdx])
            draw = ImageDraw.Draw(image, "RGBA")
            result = results[fileIdx]
            cellPositions = result[0]
            cellResults = result[1]
            for i in range(len(cellPositions)):
                if isinstance(cellResults[i], int) and cellResults[i] == -2:
                    draw.rectangle(cellPositions[i], fill=FILTER_COLOR)
            for i in range(len(cellPositions)):
                if isinstance(cellResults[i], int) and cellResults[i] == -1:
                    draw.rectangle(cellPositions[i], fill=COARSE_COLOR)
            for i in range(len(cellPositions)):
                if not isinstance(cellResults[i], int) and cellResults[i][0] > 0.5:
                    draw.rectangle(cellPositions[i], outline="black", fill=POS_COLOR)
            #save the image
            image.save(outputFilenames[fileIdx])
            print("Wrote %s" % outputFilenames[fileIdx])
