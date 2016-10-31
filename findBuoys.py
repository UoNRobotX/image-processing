import sys, re, os, argparse, time
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from constants import *
from findBuoys_input import getCellFilter, CoarseBatchProducer, BatchProducer

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
            Pre-existing files will not be replaced.
        samples file1
            Generate input samples for the detailed (or coarse) network.
                By default, the output is written to "out.jpg".
            Pre-existing files will not be replaced.
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

#constants
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
    x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_input')
    p_dropout = tf.placeholder(tf.float32, name='p_dropout')
def createCoarseNetwork(x, y_):
    #helper functions
    def createLayer(input, inSize, outSize, netName, layerName, variables, summaries):
        with tf.name_scope(layerName):
            with tf.name_scope('weights'):
                w = tf.Variable(
                    tf.truncated_normal([inSize, outSize], stddev=0.5)
                )
                variables.append(w)
                #add summaries
                mean = tf.reduce_mean(w)
                summaries.append(tf.scalar_summary(netName + '/mean/' + layerName + '/weights', mean))
                summaries.append(tf.scalar_summary(
                    netName + '/stddev/' + layerName + '/weights', tf.reduce_mean(tf.square(w - mean))
                ))
                summaries.append(tf.histogram_summary(netName + '/' + layerName + '/weights', w))
            with tf.name_scope('biases'):
                b = tf.Variable(tf.constant(0.1, shape=[outSize]))
                variables.append(b)
                #add summaries
                mean = tf.reduce_mean(b)
                summaries.append(tf.scalar_summary(netName + '/mean/' + layerName + '/biases', mean))
                summaries.append(tf.scalar_summary(
                    netName + '/stddev/' + layerName + '/biases', tf.reduce_mean(tf.square(b - mean))
                ))
                summaries.append(tf.histogram_summary(netName + '/' + layerName + '/biases', b))
            return tf.nn.sigmoid(tf.matmul(input, w) + b, 'out')
    #create nodes
    NET_NAME = 'coarse_net'
    summaries = []
    variables = []
    with tf.name_scope(NET_NAME):
        with tf.name_scope('input_reshape'):
            x_flat = x
            grayscale = True #RGB -> grayscale
            if grayscale:
                x_flat = tf.reduce_mean(x, 3)
                INPUT_CHANNELS = 1
            x_flat = tf.reshape(x_flat, [-1, INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS])
            x_flat = tf.div(x_flat, tf.constant(255.0)) #normalize values
            #add summary
            summaries.append(tf.image_summary(NET_NAME + '/input', x, 10))
        h = createLayer(
            x_flat, INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS, 30, NET_NAME, 'hidden_layer', variables, summaries
        )
        y = createLayer(
            h, 30, 1, NET_NAME, 'output_layer', variables, summaries
        )
        #y = createLayer(
        #    x_flat, INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS, 1, NET_NAME, 'output_layer', variables, summaries
        #)
        y2 = tf.slice(y_, [0, 0], [-1, 1])
        #cost
        with tf.name_scope('cost'):
            cost = tf.square(y2 - y)
            #add summary
            summaries.append(tf.scalar_summary(NET_NAME + '/cost', tf.reduce_mean(cost)))
        #optimizer
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(cost)
        #accuracy
        with tf.name_scope('accuracy'):
            y_pred = tf.greater(y, tf.constant(threshold))
            y2_pred = tf.greater(y2, tf.constant(0.5))
            correctness = tf.equal(y_pred, y2_pred)
            accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
            truePos = tf.reduce_sum(tf.cast(
                tf.logical_and(correctness, tf.equal(y_pred, tf.constant(True))),
                tf.float32
            ))
            predPos = tf.reduce_sum(tf.cast(y_pred, tf.float32))
            actualPos = tf.reduce_sum(tf.cast(y2_pred, tf.float32))
            prec = tf.cond(
                tf.equal(predPos, tf.constant(0.0)),
                lambda: tf.constant(0.0),
                lambda: truePos / predPos
            )
            rec  = tf.cond(
                tf.equal(actualPos, tf.constant(0.0)),
                lambda: tf.constant(0.0),
                lambda: truePos / actualPos
            )
            #add summary
            summaries.append(tf.scalar_summary(NET_NAME + '/accuracy', accuracy))
            summaries.append(tf.scalar_summary(NET_NAME + '/precision', prec))
            summaries.append(tf.scalar_summary(NET_NAME + '/recall', rec))
    #return output nodes and trainer
    return y, accuracy, prec, rec, train, variables, tf.merge_summary(summaries)
def createDetailedNetwork(x, y_, p_dropout):
    #helper functions
    def createWeights(shape):
        with tf.name_scope('weights'):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    def createBiases(shape):
        with tf.name_scope('biases'):
            return tf.Variable(tf.constant(0.1, shape=shape))
    def createConv(x, w, b):
        with tf.name_scope('conv'):
            xw = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
            return tf.nn.relu(xw + b)
    def createPool(c):
        with tf.name_scope('pool'):
            return tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    def addSummaries(w, b, summaries, layerName):
        #weight summaries
        wMean = tf.reduce_mean(w)
        summaries.append(tf.scalar_summary(layerName + '/mean/weights', wMean))
        summaries.append(tf.scalar_summary(
            layerName + 'stddev/weights', tf.reduce_mean(tf.square(w-wMean))
        ))
        summaries.append(tf.histogram_summary(layerName + '/weights', w))
        #biases summaries
        bMean = tf.reduce_mean(b)
        summaries.append(tf.scalar_summary(layerName + '/mean/biases', bMean))
        summaries.append(tf.scalar_summary(
            layerName + 'stddev/biases', tf.reduce_mean(tf.square(b-bMean))
        ))
        summaries.append(tf.histogram_summary(layerName + '/biases', b))
    #create nodes
    NET_NAME = 'detailed_net'
    summaries = []
    with tf.name_scope(NET_NAME):
        #add input image summary
        summaries.append(tf.image_summary(NET_NAME + '/input', x, 10))
        #first convolutional layer
        with tf.name_scope('conv_layer1'):
            w1 = createWeights([5, 5, 3, 32]) #filter_height, filter_width, in_channels, out_channels
            b1 = createBiases([32])
            c1 = createConv(x, w1, b1)
            p1 = createPool(c1)
            #addSummaries(w1, b1, summaries, NET_NAME + '/conv_layer1')
        #second convolutional layer
        with tf.name_scope('conv_layer2'):
            w2 = createWeights([5, 5, 32, 64])
            b2 = createBiases([64])
            c2 = createConv(p1, w2, b2)
            p2 = createPool(c2)
            #addSummaries(w1, b1, summaries, NET_NAME + '/conv_layer2')
        #densely connected layer
        with tf.name_scope('dense_layer'):
            w3 = createWeights([INPUT_HEIGHT//4 * INPUT_WIDTH//4 * 64, 1024])
            b3 = createBiases([1024])
            p2_flat = tf.reshape(p2, [-1, INPUT_HEIGHT//4 * INPUT_WIDTH//4 * 64])
            h1 = tf.nn.relu(tf.matmul(p2_flat, w3) + b3)
            #addSummaries(w3, b3, summaries, NET_NAME + '/dense_layer')
        #dropout
        h1_dropout = tf.nn.dropout(h1, p_dropout)
        #readout layer
        with tf.name_scope('readout_layer'):
            w4 = createWeights([1024, 2])
            b4 = createBiases([2])
            y  = tf.nn.softmax(tf.matmul(h1_dropout, w4) + b4)
        #cost
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(
                -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)),
                reduction_indices=[1])
            )
            #add summary
            summaries.append(tf.scalar_summary(NET_NAME + '/cost', cost))
        #optimizer
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(cost)
        #accuracy
        with tf.name_scope('accuracy'):
            y_pred  = tf.greater(tf.slice(y,  [0, 0], [-1, 1]), tf.slice(y,  [0, 1], [-1, 1]))
            y2_pred = tf.greater(tf.slice(y_, [0, 0], [-1, 1]), tf.slice(y_, [0, 1], [-1, 1]))
            correctness = tf.equal(y_pred, y2_pred)
            accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
            truePos = tf.reduce_sum(tf.cast(
                tf.logical_and(correctness, tf.equal(y_pred, tf.constant(True))),
                tf.float32
            ))
            predPos = tf.reduce_sum(tf.cast(y_pred, tf.float32))
            actualPos = tf.reduce_sum(tf.cast(y2_pred, tf.float32))
            prec = tf.cond(
                tf.equal(predPos, tf.constant(0.0)),
                lambda: tf.constant(0.0),
                lambda: truePos / predPos
            )
            rec  = tf.cond(
                tf.equal(actualPos, tf.constant(0.0)),
                lambda: tf.constant(0.0),
                lambda: truePos / actualPos
            )
            #add summary
            summaries.append(tf.scalar_summary(NET_NAME + '/accuracy', accuracy))
            summaries.append(tf.scalar_summary(NET_NAME + '/precision', prec))
            summaries.append(tf.scalar_summary(NET_NAME + '/recall', rec))
    #variables
    variables = [w1, b1, w2, b2, w3, b3, w4, b4]
    #return output nodes and trainer
    return y, accuracy, prec, rec, train, variables, tf.merge_summary(summaries)
cy, caccuracy, cprecision, crecall, ctrain, cvariables, csummaries = createCoarseNetwork(x, y_)
y, accuracy, precision, recall, train, variables, summaries = createDetailedNetwork(x, y_, p_dropout)

#create savers
saver = tf.train.Saver(tf.all_variables())

#use graph
with tf.Session() as sess:
    #initialising
    if os.path.exists(SAVE_FILE):
        saver.restore(sess, SAVE_FILE)
    else:
        sess.run(tf.initialize_all_variables())
    if reinitialise:
        if useCoarseOnly:
            sess.run(tf.initialize_variables(cvariables))
        else:
            sess.run(tf.initialize_variables(variables))
    #training
    if mode == "train":
        if useCoarseOnly: #train coarse network
            summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/train/coarse', sess.graph)
            prod = CoarseBatchProducer(dataFile, cellFilter)
            summariesNode = csummaries
            trainNode = ctrain
            accuracyNode = caccuracy
            precNode = cprecision
            recNode = crecall
            feedDictDefaults = {}
            feedDictDefaultsAcc = {}
        else: #train detailed network
            summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/train/detailed', sess.graph)
            prod = BatchProducer(dataFile, cellFilter, x, cy)
            summariesNode = summaries
            trainNode = train
            accuracyNode = accuracy
            precNode = precision
            recNode = recall
            feedDictDefaults = {p_dropout: 0.5}
            feedDictDefaultsAcc = {p_dropout: 1.0}
        #train
        startTime = time.time()
        for step in range(TRAINING_STEPS):
            inputs, outputs = prod.getBatch(TRAINING_BATCH_SIZE)
            feedDict = feedDictDefaults.copy()
            feedDict.update({x: inputs, y_: outputs})
            if step > 0 and step % TRAINING_RUN_PERIOD == 0: #occasionally save runtime metadata
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run(
                    [summariesNode, trainNode],
                    feed_dict=feedDict,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata
                )
                summaryWriter.add_run_metadata(run_metadata, 'step%03d' % step)
            else:
                summary, _ = sess.run(
                    [summariesNode, trainNode],
                    feed_dict=feedDict
                )
            summaryWriter.add_summary(summary, step) #write summary data for viewing with tensorboard
            #occasionally print step and accuracy
            if step % TRAINING_LOG_PERIOD == 0 or step == TRAINING_STEPS-1:
                feedDictAcc = feedDictDefaultsAcc.copy()
                feedDictAcc.update({x: inputs, y_: outputs})
                acc, prec, rec = sess.run([accuracyNode, precNode, recNode], feed_dict=feedDictAcc)
                #print("%7.2f secs - step %d, accuracy %g" % (time.time() - startTime, step, acc))
                rps = (outputs.argmax(1) == 0).sum() / len(outputs) #num positive samples / num samples
                print(
                    "%7.2f secs - step %4d, accuracy %.2f, precision %.2f, recall %.2f, rps %.2f" %
                    (time.time() - startTime, step, acc, prec, rec, rps)
                )
            #occasionally save variable values
            if step > 0 and step % TRAINING_SAVE_PERIOD == 0:
                saver.save(sess, SAVE_FILE)
        summaryWriter.close()
    #testing
    elif mode == "test":
        if useCoarseOnly: #test coarse network
            summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/test/coarse', sess.graph)
            prod = CoarseBatchProducer(dataFile, cellFilter)
            summariesNode = csummaries
            trainNode = ctrain
            accuracyNode = caccuracy
            precNode = cprecision
            recNode = crecall
            feedDictDefaults = {}
        else: #test detailed network
            summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/test/detailed', sess.graph)
            prod = BatchProducer(dataFile, cellFilter, x, cy)
            summariesNode = summaries
            trainNode = train
            accuracyNode = accuracy
            precNode = precision
            recNode = recall
            feedDictDefaults = {p_dropout: 1.0}
        #test
        startTime = time.time()
        metrics = [] #[[accuracy, precision, recall], ...]
        for step in range(TESTING_STEPS):
            inputs, outputs = prod.getBatch(TESTING_BATCH_SIZE)
            feedDict = feedDictDefaults.copy()
            feedDict.update({x: inputs, y_: outputs})
            if step > 0 and step % TESTING_RUN_PERIOD == 0: #if saving runtime metadata
                run_metadata = tf.RunMetadata()
                summary, acc, prec, rec = sess.run(
                    [summariesNode, accuracyNode, precNode, recNode],
                    feed_dict=feedDict,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata
                )
                summaryWriter.add_run_metadata(run_metadata, 'step%03d' % step)
            else:
                summary, acc, prec, rec = sess.run(
                    [summariesNode, accuracyNode, precNode, recNode],
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
    #running
    elif mode == "run":
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
                print("Invalid output directory", file=sys.stderr)
                sys.exit(1)
            outputFilenames = [outputDir + "/" + os.path.basename(name) for name in filenames]
        else:
            print("Invalid input file(s)", file=sys.stderr)
            sys.exit(1)
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
                        out = cy.eval(feed_dict={
                            x: array[
                                :,
                                INPUT_HEIGHT*i:INPUT_HEIGHT*(i+1),
                                INPUT_WIDTH*j:INPUT_WIDTH*(j+1),
                                :
                            ]
                        })
                        if useCoarseOnly:
                            p[i][j] = out[0] > threshold if thresholdGiven else out[0]
                        else:
                            if out[0] > threshold:
                                p[i][j] = -1
                            else:
                                out = y.eval(feed_dict={
                                    x: array[
                                        :,
                                        INPUT_HEIGHT*i:INPUT_HEIGHT*(i+1),
                                        INPUT_WIDTH*j:INPUT_WIDTH*(j+1),
                                        :
                                    ],
                                    p_dropout: 1.0
                                })
                                p[i][j] = out[0][0]
            #write results to image file
            draw = ImageDraw.Draw(image, "RGBA")
            for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
                for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                    if p[i][j] >= 0:
                        draw.rectangle([
                            INPUT_WIDTH*IMG_DOWNSCALE*j,
                            INPUT_HEIGHT*IMG_DOWNSCALE*i + int(INPUT_HEIGHT*IMG_DOWNSCALE*(1-p[i][j])),
                            INPUT_WIDTH*IMG_DOWNSCALE*(j+1),
                            INPUT_HEIGHT*IMG_DOWNSCALE*(i+1),
                        ], fill=(0,255,0,96))
                    elif p[i][j] == -1:
                        draw.rectangle([
                            INPUT_WIDTH*IMG_DOWNSCALE*j,
                            INPUT_HEIGHT*IMG_DOWNSCALE*i,
                            INPUT_WIDTH*IMG_DOWNSCALE*(j+1),
                            INPUT_HEIGHT*IMG_DOWNSCALE*(i+1),
                        ], fill=(196,128,0,96))
                    else:
                        draw.rectangle([
                            INPUT_WIDTH*IMG_DOWNSCALE*j,
                            INPUT_HEIGHT*IMG_DOWNSCALE*i,
                            INPUT_WIDTH*IMG_DOWNSCALE*(j+1),
                            INPUT_HEIGHT*IMG_DOWNSCALE*(i+1),
                        ], fill=(196,0,0,96))
                    draw.rectangle([
                        INPUT_WIDTH*IMG_DOWNSCALE*j,
                        INPUT_HEIGHT*IMG_DOWNSCALE*i,
                        INPUT_WIDTH*IMG_DOWNSCALE*(j+1),
                        INPUT_HEIGHT*IMG_DOWNSCALE*(i+1),
                    ], outline=(0,0,0,255))
            if not os.path.exists(outputFilenames[fileIdx]):
                image.save(outputFilenames[fileIdx])
                print(
                    "Time taken: %.2f secs, image written to %s" %
                    (time.time() - startTime, outputFilenames[fileIdx])
                )
            else:
                print(
                    "Time taken: %.2f secs, no image %s written, as it already exists" %
                    (time.time() - startTime, outputFilenames[fileIdx])
                )
    #sample generating
    elif mode == "samples":
        NUM_SAMPLES = (20, 20)
        outputImg = outputImg or "out.jpg"
        if useCoarseOnly:
            prod = CoarseBatchProducer(dataFile, cellFilter)
        else:
            prod = BatchProducer(dataFile, cellFilter, x, cy)
        image = Image.new("RGB", (INPUT_WIDTH*NUM_SAMPLES[0], INPUT_HEIGHT*NUM_SAMPLES[1]))
        draw = ImageDraw.Draw(image, "RGBA")
        #get samples
        numPositive = 0
        startTime = time.time()
        for i in range(NUM_SAMPLES[0]):
            for j in range(NUM_SAMPLES[1]):
                inputs, outputs = prod.getBatch(1)
                sampleImage = Image.fromarray(
                    inputs[0].astype(np.uint8),
                    "RGB"
                )
                image.paste(
                    sampleImage,
                    (INPUT_WIDTH*i, INPUT_HEIGHT*j, INPUT_WIDTH*(i+1), INPUT_HEIGHT*(j+1))
                )
                if useCoarseOnly:
                    isPositive = outputs[0][0] > threshold
                else:
                    isPositive = outputs[0][0] > outputs[0][1]
                if isPositive:
                    draw.rectangle([
                        INPUT_WIDTH*i,
                        INPUT_HEIGHT*j,
                        INPUT_WIDTH*(i+1),
                        INPUT_HEIGHT*(j+1),
                    ], fill=(0,255,0,64))
                    numPositive += 1
        #output time taken
        print("Time taken: %.2f secs" % (time.time() - startTime))
        print("Ratio of positive samples: %.2f" % (numPositive / (NUM_SAMPLES[0]*NUM_SAMPLES[1])))
        #save image
        if not os.path.exists(outputImg):
            image.save(outputImg)
            print("Output written to %s" % outputImg)
        else:
            print("No output file %s created, as a file already exists" % outputImg)
    #saving
    saver.save(sess, SAVE_FILE)
