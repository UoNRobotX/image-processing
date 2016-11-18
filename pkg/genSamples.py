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
        prod = CoarseBatchProducer(dataFile, cellFilter)
    else:
        prod = DetailedBatchProducer(dataFile, cellFilter)
    image = Image.new("RGB", (INPUT_WIDTH*NUM_SAMPLES[0], INPUT_HEIGHT*NUM_SAMPLES[1]))
    draw = ImageDraw.Draw(image, "RGBA")
    #get samples
    inputs, outputs = prod.getBatch(NUM_SAMPLES[0] * NUM_SAMPLES[1])
    numPositive = 0
    for idx in range(len(outputs)):
        sampleImage = Image.fromarray(inputs[idx].astype(np.uint8), "RGB")
        i = idx % NUM_SAMPLES[0]
        j = idx // NUM_SAMPLES[0]
        image.paste(
            sampleImage,
            (INPUT_WIDTH*i, INPUT_HEIGHT*j, INPUT_WIDTH*(i+1), INPUT_HEIGHT*(j+1))
        )
        #color sample green if positive
        if outputs[idx][0] > threshold:
            draw.rectangle([
                i     * INPUT_WIDTH,
                j     * INPUT_HEIGHT,
                (i+1) * INPUT_WIDTH,
                (j+1) * INPUT_HEIGHT
            ], fill=(0,255,0,64))
            numPositive += 1
    #output info
    print("Time taken: %.2f secs" % (time.time() - startTime))
    print("Ratio of positive samples: %.2f" % (numPositive / (NUM_SAMPLES[0]*NUM_SAMPLES[1])))
    #save image
    image.save(outFile)
    print("Output written to %s" % outFile)
