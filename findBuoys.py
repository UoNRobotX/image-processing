import argparse

from findBuoys_train import train
from findBuoys_test import test
from findBuoys_run import run
from findBuoys_genSamples import genSamples

#process command line arguments
description = """
    Loads/trains/tests/runs the coarse/detailed networks.
    By default, network values are loaded from files if they exist.
    By default, the detailed network is operated on.
    "mode1" specifies an action:
        train file1
            Train the detailed (or coarse) network, using training data.
        test file1
            Test detailed (or coarse) network, using testing data.
        run file1
            Run the detailed (or coarse) network on an input image.
                By default, the output is written to "out.jpg".
            A directory may be specified, in which case JPG files in it are used.
                By default, the outputs are written to same-name files.
        samples file1
            Generate input samples for the detailed (or coarse) network.
                By default, the output is written to "out.jpg".
    If operating on the detailed network, the coarse network is still used to filter input.
    If "file2" is present, it specifies a cell filter to use.
"""
parser = argparse.ArgumentParser(
    description=description,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("mode", metavar="mode1", choices=["train", "test", "run", "samples"])
parser.add_argument("file1")
parser.add_argument("file2", nargs="?")
parser.add_argument("-c", dest="useCoarseOnly", action="store_true", \
    help="Operate on the coarse network.")
parser.add_argument("-n", dest="reinitialise",  action="store_true", \
    help="Reinitialise the values of the detailed (or coarse) network.")
parser.add_argument("-s", dest="numSteps", type=int, default=100, \
    help="When training/testing, specifies the number of training/testing steps.")
parser.add_argument("-o", dest="outputImg", \
    help="When running or generating samples, specifies the output image file or directory.")
parser.add_argument("-t", dest="threshold", type=float, \
    help="Affects the precision-recall tradeoff.\
        If operating on the coarse network, positive predictions will be those above this value.\
        The default is 0.5.\
        If running on input images, causes positive prediction cells to be fully colored.")
args = parser.parse_args()

#set variables from command line arguments
mode           = args.mode
dataFile       = args.file1
filterFile     = args.file2
useCoarseOnly  = args.useCoarseOnly
reinitialise   = args.reinitialise
numSteps       = args.numSteps
outputImg      = args.outputImg
threshold      = args.threshold or 0.5
thresholdGiven = args.threshold != None

#check variables
if numSteps <= 0:
    raise Exception("Negative number of steps")
if threshold <= 0 or threshold >= 1:
    raise Exception("Invalid threshold")

#use graph
if mode == "train":
    train(dataFile, filterFile, useCoarseOnly, reinitialise, numSteps, threshold)
elif mode == "test":
    test(dataFile, filterFile, useCoarseOnly, reinitialise, numSteps, threshold)
elif mode == "run":
    run(dataFile, filterFile, useCoarseOnly, reinitialise, outputImg, threshold, thresholdGiven)
elif mode == "samples":
    genSamples(dataFile, filterFile, useCoarseOnly, outputImg, threshold)
