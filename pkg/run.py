import os, time, re
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .network import createCoarseNetwork, createDetailedNetwork

def run(dataFile, filterFile, useCoarseOnly, reinitialise, outFile, threshold, thresholdGiven):
    textOutput = [] if re.search(r"\.txt$", outFile) != None else None
    if textOutput != None and not useCoarseOnly:
        raise Exception("Text output not implemented for detailed network")
    #get input files
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
        if textOutput == None:
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
    detailedNet = None
    if not useCoarseOnly:
        detailedNet = createDetailedNetwork()
    #iterate through input images
    for fileIdx in range(len(filenames)):
        startTime = time.time()
        results = [
            [0 for j in range(IMG_WIDTH//CELL_WIDTH)]
            for i in range(IMG_HEIGHT//CELL_HEIGHT)
        ]
        staticFilteredFlag = -2
        coarseFilteredFlag = -1
        #obtain PIL image
        image = Image.open(filenames[fileIdx])
        #get cells
        cellImgs = []
        cellIndices = []
        for row in range(IMG_HEIGHT//CELL_HEIGHT):
            for col in range(IMG_WIDTH//CELL_WIDTH):
                if cellFilter[row][col] == 1:
                    #filter with static filter
                    results[row][col] = staticFilteredFlag
                else:
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
        #end pre-processing
        preProcessingTime = time.time() - startTime
        #filter with coarse network
        outputs = runNetwork(coarseNet, inputs, useCoarseOnly and reinitialise, COARSE_SAVE_FILE)
        #use detailed network if requested
        if not useCoarseOnly:
            #mark coarse filtered cells
            unfiltered = []
            for i in range(len(outputs)):
                if outputs[i][0] > threshold:
                    results[cellIndices[i][0]][cellIndices[i][1]] = coarseFilteredFlag
                else:
                    unfiltered.append(i)
            inputs = [inputs[i] for i in unfiltered]
            cellIndices = [cellIndices[i] for i in unfiltered]
            #run detailed network
            outputs = runNetwork(detailedNet, inputs, reinitialise, DETAILED_SAVE_FILE)
        #store results
        for i in range(len(outputs)):
            results[cellIndices[i][0]][cellIndices[i][1]] = outputs[i][0]
        #end processing
        processingTime = time.time() - startTime - preProcessingTime
        #output results
        if textOutput != None:
            textOutput.append(filenames[fileIdx])
            for row in results:
                line = ["1" if cell > threshold else "0" for cell in row]
                textOutput.append(" " + "".join(line))
            print("Processed %s, pre-processing time %.2f secs, processing time %.2f secs" % \
                (filenames[fileIdx], preProcessingTime, processingTime))
        else:
            #write results to image file
            FILTER_COLOR   = (128, 0, 128, 128)
            COARSE_COLOR   = (192, 160, 0, 128)
            DETAILED_COLOR = (0, 255, 0, 128)
            draw = ImageDraw.Draw(image, "RGBA")
            for i in range(IMG_HEIGHT//CELL_HEIGHT):
                for j in range(IMG_WIDTH//CELL_WIDTH):
                    #draw a rectangle, indicating confidence, coarse filtering, or filtering
                    rect = [CELL_WIDTH*j, CELL_HEIGHT*i, CELL_WIDTH*(j+1), CELL_HEIGHT*(i+1)]
                    #draw a grid cell outline
                    draw.rectangle(rect, outline=(0,0,0,255))
                    if results[i][j] >= 0:
                        if not thresholdGiven:
                            rect[1] += int(CELL_HEIGHT*(1-results[i][j]))
                            draw.rectangle(rect, fill=DETAILED_COLOR)
                        elif results[i][j] > threshold:
                            draw.rectangle(rect, fill=DETAILED_COLOR)
                    elif results[i][j] == -1:
                        draw.rectangle(rect, fill=COARSE_COLOR)
                    else:
                        draw.rectangle(rect, fill=FILTER_COLOR)
            #save the image, and print info
            image.save(outputFilenames[fileIdx])
            print("Wrote image %s, pre-processing time %.2f, processing time %.2f secs" % \
                (outputFilenames[fileIdx], preProcessingTime, processingTime))
    if textOutput != None:
        #write results to training/testing data file (used to bootstrap training)
        with open(outFile, "w") as file:
            for line in textOutput:
                file.write(line + "\n")

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
