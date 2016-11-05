import os, time

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .network import createCoarseNetwork, createDetailedNetwork, testNetwork

def test(dataFile, filterFile, useCoarseOnly, reinitialise, outFile, numSteps, threshold):
    startTime = time.time()
    #initialise
    cellFilter = getCellFilter(filterFile)
    if useCoarseOnly: #test coarse network
        net = createCoarseNetwork(threshold)
        prod = CoarseBatchProducer(dataFile, cellFilter, outFile)
        summaryDir = COARSE_SUMMARIES + "/test"
        saveFile = COARSE_SAVE_FILE
    else: #test detailed network
        net = createDetailedNetwork()
        prod = DetailedBatchProducer(dataFile, cellFilter, outFile)
        summaryDir = DETAILED_SUMMARIES + "/test"
        saveFile = DETAILED_SAVE_FILE
    print("Startup time: %.2f secs" % (time.time() - startTime))
    #test
    testNetwork(net, numSteps, prod, summaryDir, reinitialise, saveFile)
