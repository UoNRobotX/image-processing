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
            [None for j in range(IMG_WIDTH//CELL_WIDTH)]
            for i in range(IMG_HEIGHT//CELL_HEIGHT)
        ] #has elements of the form [flag, output_value]
        FLAG_FILTER       = 0
        FLAG_COARSE_NET   = 1
        FLAG_DETAILED_NET = 2
        #obtain PIL image
        image = Image.open(filenames[fileIdx])
        #get cells
        cellImgs = []
        cellIndices = []
        for row in range(IMG_HEIGHT//CELL_HEIGHT):
            for col in range(IMG_WIDTH//CELL_WIDTH):
                if cellFilter[row][col] == 1:
                    #filter with static filter
                    results[row][col] = [FLAG_FILTER, None]
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
        preProcessingTime = time.time() - startTime
        #use networks
        if useCoarseOnly:
            outputs = runNetwork(coarseNet, inputs, reinitialise, COARSE_SAVE_FILE)
            for i in range(len(outputs)):
                results[cellIndices[i][0]][cellIndices[i][1]] = [FLAG_COARSE_NET, outputs[i]]
        else:
            if True: #use coarse network
                outputs = runNetwork(coarseNet, inputs, False, COARSE_SAVE_FILE)
                #mark coarse filtered cells
                unfiltered = []
                for i in range(len(outputs)):
                    if outputs[i][0] > threshold:
                        results[cellIndices[i][0]][cellIndices[i][1]] = [FLAG_COARSE_NET, outputs[i]]
                    else:
                        unfiltered.append(i)
                inputs = [inputs[i] for i in unfiltered]
                cellIndices = [cellIndices[i] for i in unfiltered]
            #use detailed network
            outputs = runNetwork(detailedNet, inputs, reinitialise, DETAILED_SAVE_FILE)
            for i in range(len(outputs)):
                results[cellIndices[i][0]][cellIndices[i][1]] = [FLAG_DETAILED_NET, outputs[i]]
        #end processing
        processingTime = time.time() - startTime - preProcessingTime
        #output results
        if textOutput != None:
            textOutput.append(filenames[fileIdx])
            for row in results:
                line = [
                    "1" if cell[0] == FLAG_COARSE_NET and cell[1][0] > threshold else
                    "0" for cell in row
                ]
                textOutput.append(" " + "".join(line))
            print("Processed %s, pre-processing time %.2f secs, processing time %.2f secs" % \
                (filenames[fileIdx], preProcessingTime, processingTime))
        else:
            #write results to image file
            FILTER_COLOR      = (128, 0, 128, 128)
            SECONDARY_COLOR   = (192, 160, 0, 128)
            PRIMARY_COLOR_POS = (0, 255, 0, 96)
            PRIMARY_COLOR_NEG = (255, 0, 0, 96)
            draw = ImageDraw.Draw(image, "RGBA")
            for i in range(IMG_HEIGHT//CELL_HEIGHT):
                for j in range(IMG_WIDTH//CELL_WIDTH):
                    rect = [CELL_WIDTH*j, CELL_HEIGHT*i, CELL_WIDTH*(j+1), CELL_HEIGHT*(i+1)]
                    #draw a grid cell outline
                    draw.rectangle(rect, outline=(0,0,0,255))
                    #draw a rectangle indicating static/coarse_net/detailed_net filtering
                    result = results[i][j]
                    if result[0] == FLAG_FILTER:
                        draw.rectangle(rect, fill=FILTER_COLOR)
                    elif not useCoarseOnly and result[0] == FLAG_COARSE_NET:
                        draw.rectangle(rect, fill=SECONDARY_COLOR)
                    else:
                        rect1 = rect.copy()
                        rect1[1] += int(CELL_HEIGHT*(1-result[1][0]))
                        rect1[2] -= CELL_WIDTH//2
                        rect2 = rect.copy()
                        rect2[0] += CELL_WIDTH//2
                        rect2[3] -= int(CELL_HEIGHT*(1-result[1][1]))
                        draw.rectangle(rect1, fill=PRIMARY_COLOR_POS)
                        draw.rectangle(rect2, fill=PRIMARY_COLOR_NEG)
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
