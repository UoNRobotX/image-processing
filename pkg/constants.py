IMG_WIDTH    = 1280
IMG_HEIGHT   = 960
IMG_CHANNELS = 3

CELL_WIDTH  = 64 #used when marking filtered/water cells
CELL_HEIGHT = 64
NUM_BOX_TYPES = 2 #used when marking bounding boxes
BOX_COLORS = ["blue", "green"]
assert len(BOX_COLORS) == NUM_BOX_TYPES

INPUT_WIDTH  = 32
INPUT_HEIGHT = 32
COARSE_BATCH_SIZE = 1000 #the number of inputs per training/testing step
DETAILED_BATCH_SIZE = 50

TRAIN_DROPOUT = 0.9 #1.0 means no dropout
COARSE_SAVE_FILE   = "model_coarse/model.ckpt" #save/load network values to/from here
DETAILED_SAVE_FILE = "model_detailed/model.ckpt"
COARSE_SUMMARIES   = "summaries_coarse" #write summary data here, for use with tensorboard
DETAILED_SUMMARIES = "summaries_detailed"

WINDOW_SCALES = [1.0, 1.5, 2.5]
WINDOW_MIN    = [0.42, 0.44, 0.5]
WINDOW_MAX    = [0.55, 0.65, 0.68]
WINDOW_MIN_Y  = [int(m * IMG_HEIGHT) for m in WINDOW_MIN]
WINDOW_MAX_Y  = [int(m * IMG_HEIGHT) for m in WINDOW_MAX]
WINDOW_STEP_X = 5/12
WINDOW_STEP_Y = 5/12

def GET_WINDOWS():
    """ Returns a list of [topLeftX,topLeftY,bottomRightX,bottomRightY] values.
        Each value specifies a window in the image to be sent to the detailed network.
    """
    cells = []
    for i in range(len(WINDOW_SCALES)):
        scale = WINDOW_SCALES[i]
        cellHeight = int(INPUT_HEIGHT * scale)
        cellWidth  = int(INPUT_WIDTH * scale)
        cell = [0, WINDOW_MAX_Y[i] - cellHeight, cellWidth, WINDOW_MAX_Y[i]]
        while True:
            if cell[2] > IMG_WIDTH:
                cell[0] = 0
                cell[2] = cellWidth
                cell[1] -= int(cellHeight * WINDOW_STEP_Y)
                cell[3] -= int(cellHeight * WINDOW_STEP_Y)
            if cell[1] < WINDOW_MIN_Y[i]:
                break
            #store cell
            cells.append(cell.copy())
            #move right
            cell[0] += int(cellWidth * WINDOW_STEP_X)
            cell[2] += int(cellWidth * WINDOW_STEP_X)
    return cells

#def GET_WINDOW(x, y):
#    """ Returns a window with some location and size, given a center """
#    if 0 <= x < IMG_WIDTH and VAR_CELL_MIN_Y <= y <= VAR_CELL_MAX_Y:
#        width = 32
#        #width = ((y-400)*(175/300) + 15) / 2
#        #width = ((y-400)*(y-400)*(200/(300*300)) + 25) / 2
#        return (int(x-width), int(y-width), int(x+width), int(y+width))
#    else:
#        return None
