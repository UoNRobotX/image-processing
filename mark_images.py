import sys, os, argparse

from pkg import mark_input
from pkg import mark_window

#process command line arguments
description = """
    First, obtains a list of image filenames.
    By default, the filenames are read from stdin, with 1 filename per line.
        Leading and trailing whitespace, empty names, and names with commas, are ignored.

    Then, each image is displayed, and the user may mark them using the mouse.
    Pressing right/left causes the next/previous image to be displayed.
    Pressing escape exits the application.
    By default, information about the markings is written to stdout.

    "mode1" may be one of the following:
        filter
            The user marks grid cells to be ignored (camera boundaries, roof, etc).
            Clicking or dragging over a cell toggles whether it is marked.
            The output contains a line for each row of cells.
                " 0111" specifies 4 cells of a row, 3 of which are marked.
        coarse
            The user marks grid cells that contain only water.
            Clicking or dragging over a cell toggles whether it is marked.
            The output contains sections, each describing cells to ignore for an image.
                Each section starts with a line containing the image filename.
                Each such line is followed by an indented line for each row.
                    " 0111" specifies 4 cells of a row, 3 of which are marked.
        detailed
            The user marks bounding boxes by clicking and dragging.
            Boxes can be deleted by right-clicking.
            The output contains lines holding image filenames.
            The output contains sections, each describing boxes for an image.
                Each section starts with a line containing the image filename.
                Each such line is followed by indented lines.
                    " 1,2,3,4" specifies a box with top-left 1,2 and bottom-right 3,4.
"""
parser = argparse.ArgumentParser(
    description=description,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("mode", metavar="mode1", choices=["filter", "coarse", "detailed"])
parser.add_argument("-d", dest="inputDir",   help="Use JPG files in a directory as the images to mark.")
parser.add_argument("-o", dest="outputFile", help="Write output to a file instead of to stdout.")
parser.add_argument("-l", dest="loadFile",   help="Load mark data from a file.")
parser.add_argument("-g", dest="skipFile",   help="Skip to a file in the list.")
parser.add_argument("-s", dest="saveDir",    help="Save the images, with markings, to a directory.")
args = parser.parse_args()

#set variables from command line arguments
mode       = args.mode
inputDir   = args.inputDir
loadFile   = args.loadFile
outputFile = args.outputFile
skipFile   = args.skipFile
saveDir    = args.saveDir

#check variables
if saveDir != None and not os.path.isdir(saveDir):
    raise Exception("Invalid output save directory")

#read input filenames, and load mark data
cellFilter = None
fileMarks = mark_input.getFilenames(inputDir)
if loadFile != None:
    if mode == "filter":
        cellFilter = mark_input.loadFilterData(loadFile)
    elif mode == "coarse":
        fileMarks = mark_input.loadCoarseSet(loadFile, fileMarks)
    elif mode == "detailed":
        fileMarks = mark_input.loadDetailedSet(loadFile, fileMarks)
if len(fileMarks) == 0:
    raise Exception("No input files")

#create window
window = mark_window.Window(mode, skipFile, cellFilter, fileMarks, outputFile, saveDir)
