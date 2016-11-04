import time
import numpy as np
from PIL import Image, ImageDraw

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer

NUM_SAMPLES = (20, 20)

def genSamples(dataFile, filterFile, useCoarseOnly, outFile, threshold):
    startTime = time.time()
    #initialise
    cellFilter = getCellFilter(filterFile)
    if useCoarseOnly:
        prod = CoarseBatchProducer(dataFile, cellFilter, None)
    else:
        prod = DetailedBatchProducer(dataFile, cellFilter, None)
    image = Image.new("RGB", (INPUT_WIDTH*NUM_SAMPLES[0], INPUT_HEIGHT*NUM_SAMPLES[1]))
    draw = ImageDraw.Draw(image, "RGBA")
    #get samples
    numPositive = 0
    for i in range(NUM_SAMPLES[0]):
        for j in range(NUM_SAMPLES[1]):
            inputs, outputs = prod.getBatch(1)
            sampleImage = Image.fromarray(inputs[0].astype(np.uint8), "RGB")
            image.paste(
                sampleImage,
                (INPUT_WIDTH*i, INPUT_HEIGHT*j, INPUT_WIDTH*(i+1), INPUT_HEIGHT*(j+1))
            )
            if useCoarseOnly:
                isPositive = outputs[0][0] > threshold
            else:
                isPositive = outputs[0][0] > outputs[0][1]
            #color sample green if positive
            if isPositive:
                draw.rectangle([
                    INPUT_WIDTH*i,
                    INPUT_HEIGHT*j,
                    INPUT_WIDTH*(i+1),
                    INPUT_HEIGHT*(j+1),
                ], fill=(0,255,0,64))
                numPositive += 1
    #output info
    print("Time taken: %.2f secs" % (time.time() - startTime))
    print("Ratio of positive samples: %f" % (numPositive / (NUM_SAMPLES[0]*NUM_SAMPLES[1])))
    #save image
    image.save(outFile)
    print("Output written to %s" % outFile)
