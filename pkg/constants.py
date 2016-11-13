IMG_HEIGHT   = 960
IMG_WIDTH    = 1280
IMG_CHANNELS = 3

CELL_HEIGHT = 64
CELL_WIDTH  = 64

INPUT_HEIGHT = CELL_HEIGHT // 2
INPUT_WIDTH  = CELL_WIDTH // 2
BATCH_SIZE   = 50 #the number of inputs per training/testing step

COARSE_SAVE_FILE   = "model_coarse/model.ckpt"   #save/load network values to/from here
DETAILED_SAVE_FILE = "model_detailed/model.ckpt"
COARSE_SUMMARIES   = "summaries_coarse" #write summary data here, for use with tensorboard
DETAILED_SUMMARIES = "summaries_detailed"

VAR_CELL_MIN_Y  = 450
VAR_CELL_MAX_Y  = 700
VAR_CELL_STEP_X = 1/2
VAR_CELL_STEP_Y = 1/2

def GET_VAR_CELL(x, y):
    if 0 <= x < IMG_WIDTH and VAR_CELL_MIN_Y <= y <= VAR_CELL_MAX_Y:
        #width = 32
        width = ((y-400)*(175/(300)) + 15) / 2
        #width = ((y-400)*(y-400)*(200/(300*300)) + 25) / 2
        return (int(x-width), int(y-width), int(x+width), int(y+width))
    else:
        return None

def GET_VAR_CELLS():
    cells = []
    center = [0, VAR_CELL_MAX_Y]
    while True:
        cell = GET_VAR_CELL(center[0], center[1])
        if cell == None:
            break
        if cell[0] < 0: #cell collides with left side
            #move right
            center[0] += -cell[0]
            if center[0] >= IMG_WIDTH:
                #move up and left
                center[0] = 0
                center[1] -= int((cell[3]-cell[1]) * VAR_CELL_STEP_Y)
            continue
        if cell[2] > IMG_WIDTH or cell[3] > IMG_HEIGHT: #cell collides with right/bottom side
            #move up and left
            center[0] = 0
            center[1] -= int((cell[3]-cell[1]) * VAR_CELL_STEP_Y)
            continue
        if cell[1] < 0: #cell collides with top side
            break
        #store cell
        cells.append(cell)
        #move right
        center[0] += int((cell[2]-cell[0]) * VAR_CELL_STEP_X)
        if center[0] >= IMG_WIDTH:
            #move up and left
            center[0] = 0
            center[1] -= int((cell[3]-cell[1]) * VAR_CELL_STEP_Y)
    return cells
