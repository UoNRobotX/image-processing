IMG_HEIGHT           = 960
IMG_WIDTH            = 1280
IMG_CHANNELS         = 3
IMG_DOWNSCALE        = 2
IMG_SCALED_HEIGHT    = IMG_HEIGHT // IMG_DOWNSCALE
IMG_SCALED_WIDTH     = IMG_WIDTH  // IMG_DOWNSCALE

INPUT_HEIGHT         = 32
INPUT_WIDTH          = 32
BATCH_SIZE           = 50 #the number of inputs per training/testing step

COARSE_SAVE_FILE     = "model_coarse/model.ckpt"   #save/load network values to/from here
DETAILED_SAVE_FILE   = "model_detailed/model.ckpt"
COARSE_SUMMARIES     = "summaries_coarse" #write summary data here, for use with tensorboard
DETAILED_SUMMARIES   = "summaries_detailed"
