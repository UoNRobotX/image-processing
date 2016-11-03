import sys, re, os, argparse, time
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from constants import *
from findBuoys_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from findBuoys_net import createCoarseNetwork, createDetailedNetwork

#process command line arguments
description = """
    Loads/trains/tests/runs the coarse/detailed networks.
    By default, network values are loaded from files if they exist.
    By default, the detailed network is operated on.
    'mode1' specifies an action:
        train file1
            Train the detailed (or coarse) network, using training data.
        test file1
            Test detailed (or coarse) network, using testing data.
        run file1
            Run the detailed (or coarse) network on an input image.
                By default, the output is written to "out.jpg".
            A directory may be specified, in which case JPG files in it are used.
                By default, the outputs are written to same-name files.
                Pre-existing files in the directory will not be replaced.
        samples file1
            Generate input samples for the detailed (or coarse) network.
                By default, the output is written to "out.jpg".
    If operating on the detailed network, the coarse network is still used to filter input.
    If 'file2' is present, it specifies a cell filter to use.
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

#runtime constants
SAVE_FILE            = "modelData/model.ckpt" #save/load network values to/from here
TRAINING_STEPS       = numSteps
TRAINING_BATCH_SIZE  = 50 #the number of inputs per training step
TRAINING_LOG_PERIOD  = 50 #informative lines are printed after this many training steps
TRAINING_SAVE_PERIOD = 1000 #save every N steps
TRAINING_RUN_PERIOD  = 50 #save runtime metadata every N steps
TESTING_STEPS        = numSteps #number of batches used for testing
TESTING_BATCH_SIZE   = 50
TESTING_LOG_PERIOD   = 10
TESTING_RUN_PERIOD   = 10
SUMMARIES_DIR        = 'summaries'

#obtain cell filter
cellFilter = getCellFilter(filterFile)

#create computation graph
with tf.name_scope('input'): #group nodes for easier viewing with tensorboard
    #create input nodes
    x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_input')
    p_dropout = tf.placeholder(tf.float32, name='p_dropout')
coarseNodes = createCoarseNetwork(x, y_, threshold)
detailedNodes = createDetailedNetwork(x, y_, p_dropout)

#create savers
saver = tf.train.Saver(tf.all_variables())

#helper functions
def train():
    startTime = time.time()
    if useCoarseOnly: #train coarse network
        nodes = coarseNodes
        prod = CoarseBatchProducer(dataFile, cellFilter)
        summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/train/coarse', sess.graph)
        feedDictDefaults = {}
        feedDictDefaultsAcc = {}
    else: #train detailed network
        nodes = detailedNodes
        prod = DetailedBatchProducer(dataFile, cellFilter, x, coarseNodes.y, threshold)
        summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/train/detailed', sess.graph)
        feedDictDefaults = {p_dropout: 0.5}
        feedDictDefaultsAcc = {p_dropout: 1.0}
    #start training
    for step in range(TRAINING_STEPS):
        inputs, outputs = prod.getBatch(TRAINING_BATCH_SIZE)
        feedDict = feedDictDefaults.copy()
        feedDict.update({x: inputs, y_: outputs})
        if step > 0 and step % TRAINING_RUN_PERIOD == 0: #occasionally save runtime metadata
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run(
                [nodes.summaries, nodes.train],
                feed_dict=feedDict,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata
            )
            summaryWriter.add_run_metadata(run_metadata, 'step%03d' % step)
        else:
            summary, _ = sess.run(
                [nodes.summaries, nodes.train],
                feed_dict=feedDict
            )
        summaryWriter.add_summary(summary, step) #write summary data for viewing with tensorboard
        #occasionally print step and accuracy
        if step % TRAINING_LOG_PERIOD == 0 or step == TRAINING_STEPS-1:
            feedDictAcc = feedDictDefaultsAcc.copy()
            feedDictAcc.update({x: inputs, y_: outputs})
            acc, prec, rec = sess.run(
                [nodes.accuracy, nodes.precision, nodes.recall],
                feed_dict=feedDictAcc
            )
            rps = (outputs.argmax(1) == 0).sum() / len(outputs) #num positive samples / num samples
            print(
                "%7.2f secs - step %4d, accuracy %.2f, precision %.2f, recall %.2f, rps %.2f" %
                (time.time() - startTime, step, acc, prec, rec, rps)
            )
        #occasionally save variable values
        if step > 0 and step % TRAINING_SAVE_PERIOD == 0:
            saver.save(sess, SAVE_FILE)
    summaryWriter.close()
def test():
    startTime = time.time()
    if useCoarseOnly: #test coarse network
        nodes = coarseNodes
        prod = CoarseBatchProducer(dataFile, cellFilter)
        summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/test/coarse', sess.graph)
        feedDictDefaults = {}
    else: #test detailed network
        nodes = detailedNodes
        prod = DetailedBatchProducer(dataFile, cellFilter, x, coarseNodes.y, threshold)
        summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/test/detailed', sess.graph)
        feedDictDefaults = {p_dropout: 1.0}
    #test
    metrics = [] #[[accuracy, precision, recall], ...]
    for step in range(TESTING_STEPS):
        inputs, outputs = prod.getBatch(TESTING_BATCH_SIZE)
        feedDict = feedDictDefaults.copy()
        feedDict.update({x: inputs, y_: outputs})
        if step > 0 and step % TESTING_RUN_PERIOD == 0: #if saving runtime metadata
            run_metadata = tf.RunMetadata()
            summary, acc, prec, rec = sess.run(
                [nodes.summaries, nodes.accuracy, nodes.precision, nodes.recall],
                feed_dict=feedDict,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata
            )
            summaryWriter.add_run_metadata(run_metadata, 'step%03d' % step)
        else:
            summary, acc, prec, rec = sess.run(
                [nodes.summaries, nodes.accuracy, nodes.precision, nodes.recall],
                feed_dict=feedDict
            )
        summaryWriter.add_summary(summary, step)
        if step % TESTING_LOG_PERIOD == 0:
            print(
                "%7.2f secs - step %4d, accuracy %.2f, precision %.2f, recall %.2f" %
                (time.time()-startTime, step, acc, prec, rec)
            )
            metrics.append([acc, prec, rec])
    accs  = [m[0] for m in metrics]
    precs = [m[1] for m in metrics]
    recs  = [m[2] for m in metrics]
    print(
        "Averages: accuracy %.2f, precision %.2f, recall %.2f" %
        (sum(accs)/len(accs), sum(precs)/len(precs), sum(recs)/len(recs))
    )
    summaryWriter.close()
def run():
    #get input files
    if os.path.isfile(dataFile):
        filenames = [dataFile]
        outputFilenames = [outputImg or "out.jpg"]
    elif os.path.isdir(dataFile):
        filenames = [
            dataFile + "/" + name for
            name in os.listdir(dataFile) if
            os.path.isfile(dataFile + "/" + name) and re.fullmatch(r".*\.jpg", name)
        ]
        filenames.sort()
        outputDir = outputImg or dataFile
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        elif not os.path.isdir(outputDir):
            raise Exception("Invalid output directory")
        outputFilenames = [outputDir + "/" + os.path.basename(name) for name in filenames]
    else:
        raise Exception("Invalid input file")
    #iterate through input images
    for fileIdx in range(len(filenames)):
        #obtain PIL image
        image = Image.open(filenames[fileIdx])
        image_scaled = image.resize((IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT), resample=Image.LANCZOS)
        #obtain numpy array
        array = np.array(list(image_scaled.getdata())).astype(np.float32)
        array = array.reshape((IMG_SCALED_HEIGHT, IMG_SCALED_WIDTH, IMG_CHANNELS))
        array = np.array([array])
        #variable for storing results
        p = [
            [0 for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH)]
            for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT)
        ]
        #get results
        startTime = time.time()
        for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
            for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                if cellFilter != None and cellFilter[i][j] == 1:
                    p[i][j] = -2
                else:
                    d = array[:, INPUT_HEIGHT*i:INPUT_HEIGHT*(i+1), INPUT_WIDTH*j:INPUT_WIDTH*(j+1), :]
                    out = coarseNodes.y.eval(feed_dict={x: d})
                    if useCoarseOnly:
                        p[i][j] = out[0] > threshold if thresholdGiven else out[0]
                    else:
                        if out[0] > threshold:
                            p[i][j] = -1
                        else:
                            out = detailedNodes.y.eval(feed_dict={x: d, p_dropout: 1.0})
                            p[i][j] = out[0][0]
        #write results to image file
        draw = ImageDraw.Draw(image, "RGBA")
        for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
            for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                #draw a rectangle, indicating confidence, coarse filtering, or filtering
                rect = [
                    INPUT_WIDTH*IMG_DOWNSCALE*j,
                    INPUT_HEIGHT*IMG_DOWNSCALE*i,
                    INPUT_WIDTH*IMG_DOWNSCALE*(j+1),
                    INPUT_HEIGHT*IMG_DOWNSCALE*(i+1),
                ]
                #draw a grid cell outline
                draw.rectangle(rect, outline=(0,0,0,255))
                if p[i][j] >= 0:
                    rect[1] += int(INPUT_HEIGHT*IMG_DOWNSCALE*(1-p[i][j]))
                    draw.rectangle(rect, fill=(0,255,0,96))
                elif p[i][j] == -1:
                    draw.rectangle(rect, fill=(196,128,0,96))
                else:
                    draw.rectangle(rect, fill=(196,0,0,96))
        #save the image, and print info
        if len(outputFilenames) > 1 and os.path.exists(outputFilenames[fileIdx]):
            print("Time taken: %.2f secs, no image %s written, as such a file already exists" % \
                (time.time() - startTime, outputFilenames[fileIdx]))
        else:
            image.save(outputFilenames[fileIdx])
            print("Time taken: %.2f secs, image written to %s" % \
                (time.time() - startTime, outputFilenames[fileIdx]))
def genSamples(outputImg):
    NUM_SAMPLES = (20, 20)
    startTime = time.time()
    if useCoarseOnly:
        prod = CoarseBatchProducer(dataFile, cellFilter)
    else:
        prod = DetailedBatchProducer(dataFile, cellFilter, x, coarseNodes.y, threshold)
    image = Image.new("RGB", (INPUT_WIDTH*NUM_SAMPLES[0], INPUT_HEIGHT*NUM_SAMPLES[1]))
    draw = ImageDraw.Draw(image, "RGBA")
    #get samples
    numPositive = 0
    for i in range(NUM_SAMPLES[0]):
        for j in range(NUM_SAMPLES[1]):
            inputs, outputs = prod.getBatch(1)
            sampleImage = Image.fromarray(inputs[0].astype(np.uint8), "RGB")
            image.paste(
                sampleImage,
                (INPUT_WIDTH*i, INPUT_HEIGHT*j, INPUT_WIDTH*(i+1), INPUT_HEIGHT*(j+1))
            )
            if useCoarseOnly:
                isPositive = outputs[0][0] > threshold
            else:
                isPositive = outputs[0][0] > outputs[0][1]
            #color sample green if positive
            if isPositive:
                draw.rectangle([
                    INPUT_WIDTH*i,
                    INPUT_HEIGHT*j,
                    INPUT_WIDTH*(i+1),
                    INPUT_HEIGHT*(j+1),
                ], fill=(0,255,0,64))
                numPositive += 1
    #output info
    print("Time taken: %.2f secs" % (time.time() - startTime))
    print("Ratio of positive samples: %.2f" % (numPositive / (NUM_SAMPLES[0]*NUM_SAMPLES[1])))
    #save image
    image.save(outputImg)
    print("Output written to %s" % outputImg)

#use graph
with tf.Session() as sess:
    #initialise
    if os.path.exists(SAVE_FILE):
        saver.restore(sess, SAVE_FILE)
    else:
        sess.run(tf.initialize_all_variables())
    if reinitialise:
        if useCoarseOnly:
            sess.run(tf.initialize_variables(coarseNodes.variables))
        else:
            sess.run(tf.initialize_variables(detailedNodes.variables))
    #use network
    if mode == "train":
        train()
    elif mode == "test":
        test()
    elif mode == "run":
        run()
    elif mode == "samples":
        genSamples(outputImg or "out.jpg")
    #save
    saver.save(sess, SAVE_FILE)
