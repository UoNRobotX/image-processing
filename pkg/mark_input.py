import sys, os, re

def getFilenames(inputDir):
    """ Obtain filenames of images to be marked, from stdin or a directory.
        Returns a dict that maps each filename to None.
    """
    fileMarks = dict()
    if inputDir == None:
        #get filenames from standard input
        for line in sys.stdin:
            line = line.strip()
            if len(line) > 0 and line.find(",") == -1:
                fileMarks[line] = None
    else:
        #get filenames from JPG files in the specified directory
        fileMarks = {
            (inputDir + "/" + name) : None for
            name in os.listdir(inputDir) if
            os.path.isfile(inputDir + "/" + name) and re.fullmatch(r".*\.jpg", name)
        }
    return fileMarks

def loadFilterData(loadFile):
    """ Loads a cell filter from a file.
        Returns a list with the form [[0, 1, ...], ...].
            Each element denotes a row of cells, where 1 indicates a filtered cell.
    """
    cellFilter = []
    with open(loadFile) as file:
        for line in file:
            cellFilter.append([int(c) for c in line.strip()])
    return cellFilter

def loadCoarseSet(loadFile, fileMarks):
    """ Reads coarse set data from "loadFile", adding it to "fileMarks".
        "fileMarks" should map filenames to mark information.
            fileMarks[filename] may be None, indicating no mark information.
            fileMarks[filename] may have the form [[0, 1, ...], ...].
                Each element denotes a row of cells, where 1 indicates a cell containing only water.
    """
    filename = None
    with open(loadFile) as file:
        for line in file:
            if line[0] != " ":
                filename = line.strip()
                fileMarks[filename] = []
            elif filename != None:
                fileMarks[filename].append([int(c) for c in line.strip()])
            else:
                raise Exception("Invalid coarse set file format")
    return fileMarks

def loadDetailedSet(loadFile, fileMarks):
    """ Reads detailed set data from "loadFile", adding it to "fileMarks".
        "fileMarks" should map filenames to mark information.
            fileMarks[filename] may be None, indicating no mark information.
            fileMarks[filename] may have the form [[x,y,x,y], ...].
                Each element denotes a bounding box's top-left and bottom-right.
    """
    filename = None
    with open(loadFile) as file:
        for line in file:
            if line[0] != " ":
                filename = line.strip()
                fileMarks[filename] = []
            elif filename != None:
                fileMarks[filename].append([int(c) for c in line.strip().split(",")])
            else:
                raise Exception("Invalid detailed set file format")
    return fileMarks
