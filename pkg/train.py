import os, re, time

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .network import createCoarseNetwork, createDetailedNetwork, trainNetwork

def train(dataFile, dataFile2, filterFile, useCoarseOnly, reinitialise, outFile, numSteps, threshold):
    startTime = time.time()
    #initialise
    cellFilter = getCellFilter(filterFile)
    if useCoarseOnly: #train coarse network
        net = createCoarseNetwork(threshold)
        prod = CoarseBatchProducer(dataFile, cellFilter, outFile and outFile + "_train")
        testProd = CoarseBatchProducer(dataFile2, cellFilter, outFile and outFile + "_validate")
        summaryDir = COARSE_SUMMARIES + "/train"
        testSummaryDir = COARSE_SUMMARIES + "/validate"
        saveFile = COARSE_SAVE_FILE
    else: #train detailed network
        net = createDetailedNetwork()
        prod = DetailedBatchProducer(dataFile, cellFilter, outFile and outFile + "_train")
        testProd = DetailedBatchProducer(dataFile2, cellFilter, outFile and outFile + "_validate")
        summaryDir = DETAILED_SUMMARIES + "/train"
        testSummaryDir = DETAILED_SUMMARIES + "/validate"
        saveFile = DETAILED_SAVE_FILE
    print("Startup time: %.2f secs" % (time.time() - startTime))
    #train
    trainNetwork(net, numSteps, prod, testProd, summaryDir, testSummaryDir, reinitialise, saveFile)
