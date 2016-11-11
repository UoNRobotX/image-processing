IMG_HEIGHT   = 960
IMG_WIDTH    = 1280
IMG_CHANNELS = 3

CELL_HEIGHT = 64
CELL_WIDTH  = 64

INPUT_HEIGHT = CELL_HEIGHT // 2
INPUT_WIDTH = CELL_WIDTH // 2
BATCH_SIZE  = 50 #the number of inputs per training/testing step

COARSE_SAVE_FILE   = "model_coarse/model.ckpt"   #save/load network values to/from here
DETAILED_SAVE_FILE = "model_detailed/model.ckpt"
COARSE_SUMMARIES   = "summaries_coarse" #write summary data here, for use with tensorboard
DETAILED_SUMMARIES = "summaries_detailed"
