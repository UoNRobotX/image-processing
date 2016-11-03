import os, time, re
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .networks import createCoarseNetwork, createDetailedNetwork

def run(dataFile, filterFile, useCoarseOnly, reinitialise, outputImg, threshold, thresholdGiven):
    startTime = time.time()
    #get input files
    if os.path.isfile(dataFile):
        filenames = [dataFile]
        outputFilenames = [outputImg or "out.jpg"]
    elif os.path.isdir(dataFile):
        filenames = [
            dataFile + "/" + name for
            name in os.listdir(dataFile) if
            os.path.isfile(dataFile + "/" + name) and re.fullmatch(r".*\.jpg", name)
        ]
        filenames.sort()
        outputDir = outputImg or dataFile
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
        image_scaled = image.resize((IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT), resample=Image.LANCZOS)
        #obtain numpy array
        array = np.array(list(image_scaled.getdata())).astype(np.float32)
        array = array.reshape((IMG_SCALED_HEIGHT, IMG_SCALED_WIDTH, IMG_CHANNELS))
        array = np.array([array])
        #variable for storing results
        p = [
            [0 for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH)]
            for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT)
        ]
        #filter with static filter
        for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
            for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                if cellFilter != None and cellFilter[i][j] == 1:
                    p[i][j] = -2
        #filter with coarse network
        with tf.Session(graph=coarseNet.graph) as sess:
            #reinitialise or load values
            if reinitialise or not os.path.exists(COARSE_SAVE_FILE):
                sess.run(tf.initialize_all_variables())
            else:
                tf.train.Saver(tf.all_variables()).restore(sess, COARSE_SAVE_FILE)
            for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
                for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                    if p[i][j] == 0:
                        d = array[:, INPUT_HEIGHT*i:INPUT_HEIGHT*(i+1), \
                            INPUT_WIDTH*j:INPUT_WIDTH*(j+1), :]
                        out = coarseNet.y.eval(feed_dict={coarseNet.x: d, coarseNet.p_dropout: 1.0})
                        if useCoarseOnly:
                            p[i][j] = out[0] > threshold if thresholdGiven else out[0]
                        elif out[0] > threshold:
                            p[i][j] = -1
        if not useCoarseOnly:
            #use detailed network
            with tf.Session(graph=detailedNet.graph) as sess:
                #reinitialise or load values
                if reinitialise or not os.path.exists(DETAILED_SAVE_FILE):
                    sess.run(tf.initialize_all_variables())
                else:
                    tf.train.Saver(tf.all_variables()).restore(sess, DETAILED_SAVE_FILE)
                for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
                    for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                        if p[i][j] == 0:
                            d = array[:, INPUT_HEIGHT*i:INPUT_HEIGHT*(i+1), \
                                INPUT_WIDTH*j:INPUT_WIDTH*(j+1), :]
                            out = detailedNet.y.eval(feed_dict={
                                detailedNet.x: d, detailedNet.p_dropout: 1.0
                            })
                            p[i][j] = out[0][0]
        #write results to image file
        draw = ImageDraw.Draw(image, "RGBA")
        for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
            for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                #draw a rectangle, indicating confidence, coarse filtering, or filtering
                rect = [
                    INPUT_WIDTH*IMG_DOWNSCALE*j,
                    INPUT_HEIGHT*IMG_DOWNSCALE*i,
                    INPUT_WIDTH*IMG_DOWNSCALE*(j+1),
                    INPUT_HEIGHT*IMG_DOWNSCALE*(i+1),
                ]
                #draw a grid cell outline
                draw.rectangle(rect, outline=(0,0,0,255))
                if p[i][j] >= 0:
                    rect[1] += int(INPUT_HEIGHT*IMG_DOWNSCALE*(1-p[i][j]))
                    draw.rectangle(rect, fill=(0,255,0,96))
                elif p[i][j] == -1:
                    draw.rectangle(rect, fill=(196,128,0,96))
                else:
                    draw.rectangle(rect, fill=(196,0,0,96))
        #save the image, and print info
        image.save(outputFilenames[fileIdx])
        print("Time taken: %.2f secs, image written to %s" % \
            (time.time() - startTime, outputFilenames[fileIdx]))
