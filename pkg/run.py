import os, time, re
import numpy as np
from PIL import Image, ImageDraw

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .network import createCoarseNetwork, createDetailedNetwork, runNetwork

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
        #obtain PIL image
        image = Image.open(filenames[fileIdx])
        #start processing
        startTime = time.time()
        #get cells
        cellImages = []
        for row in range(IMG_HEIGHT//CELL_HEIGHT):
            cellImages.append([])
            for col in range(IMG_WIDTH//CELL_WIDTH):
                cellImg = image.crop(
                    (col*CELL_WIDTH, row*CELL_HEIGHT, (col+1)*CELL_WIDTH, (row+1)*CELL_HEIGHT)
                )
                cellImg = cellImg.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.LANCZOS)
                cellImages[-1].append(cellImg)
        #obtain numpy arrays
        cellData = [
            [
                np.array(list(cellImages[row][col].getdata())).astype(np.float32).reshape(
                    (INPUT_HEIGHT, INPUT_WIDTH, IMG_CHANNELS)
                ) for col in range(len(cellImages[row]))
            ]
            for row in range(len(cellImages))
        ]
        #used for storing results
        results = [
            [0 for j in range(IMG_WIDTH//CELL_WIDTH)]
            for i in range(IMG_HEIGHT//CELL_HEIGHT)
        ]
        staticFilteredFlag = -2
        coarseFilteredFlag = -1
        #filter with static filter
        for i in range(IMG_HEIGHT//CELL_HEIGHT):
            for j in range(IMG_WIDTH//CELL_WIDTH):
                if cellFilter != None and cellFilter[i][j] == 1:
                    results[i][j] = staticFilteredFlag
        #filter with coarse network
        runNetwork(coarseNet, cellData, results, useCoarseOnly and reinitialise, COARSE_SAVE_FILE)
        #use detailed network if requested
        if not useCoarseOnly:
            #mark coarse filtered cells
            for i in range(len(results)):
                for j in range(len(results[i])):
                    if results[i][j] > threshold:
                        results[i][j] = coarseFilteredFlag
            #run detailed network
            runNetwork(detailedNet, cellData, results, reinitialise, DETAILED_SAVE_FILE)
        #end processing
        processingTime = time.time() - startTime
        #output results
        if textOutput != None:
            textOutput.append(filenames[fileIdx])
            for row in results:
                line = ["1" if cell > threshold else "0" for cell in row]
                textOutput.append(" " + "".join(line))
            print("Processed %s, processing time %.2f secs" % \
                (filenames[fileIdx], processingTime))
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
            print("Wrote image %s, processing time %.2f secs" % \
                (outputFilenames[fileIdx], processingTime))
    if textOutput != None:
        #write results to training/testing data file (used to bootstrap training)
        with open(outFile, "w") as file:
            for line in textOutput:
                file.write(line + "\n")
