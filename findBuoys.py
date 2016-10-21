import sys, os, math, random, time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import tensorflow as tf

usage = "Usage: python3 " + sys.argv[0] + """ cmd1 file1 [file2] [-cdn] [-s n1] [-o img1]
    Loads/trains/tests/runs the coarse/detailed networks.
    By default, network values are loaded from files if they exist.
    If neither -c nor -d is given, -d is assumed.

    'cmd1' specifies an action:
        train file1 [file2]
            Train the coarse/detailed network, using training data from 'file1'.
            The filter specified by 'file2' is used (if empty, no filter is used).
        test file1 [file2]
            Test coarse/detailed networks, using testing data from 'file1'.
            The filter specified by 'file2' is used (if empty, no filter is used).
        run file1 [file2]
            Run coarse/detailed networks using input image 'file1'.
            The filter specified by 'file2' is used (if empty, no filter is used).
        samples file1 [file2]
            Generate input samples for the coarse/detailed network, using data from file 'f1'.
    For each action, the filter specified by 'file2' is used (default is 'filterData.txt')
    If 'file2' is empty, no filter is used.

    Options:
        -c
            With 'train', train the coarse network.
            With 'test', test the coarse network, ignoring the detailed network.
            With 'run', run the coarse network on the image, ignoring the detailed network.
            With 'samples', generate inputs to the coarse network.
            With -n, only re-initialise values for the coarse network.
        -d
            With 'train', train the detailed network.
            With 'test', test the detailed network.
            With 'run', run the detailed network on the image.
            With 'samples', generate inputs to the detailed network.
            With -n, only re-initialise values for the detailed network.
            The coarse network is still used to filter input.
        -n
            Re-initialise values for the coarse/detailed network.
        -s n1
            With 'train' or 'test', specifies the number of steps done.
        -o img1
            With 'run' or 'sampels', specifies the image file to create.
            The defaults are 'output.jpg' and 'samples.jpg'.
"""

#process command line arguments
MODE_TRAIN    = 0
MODE_TEST     = 1
MODE_RUN      = 2
MODE_SAMPLES  = 3
mode          = None
useCoarseOnly = False
dataFile      = None
filterFile    = None
outputImg     = None
reinitialise  = False
numSteps = 100
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "-c":
        useCoarseOnly = True
    elif arg == "-d":
        useCoarseOnly = False
    elif arg == "-n":
        reinitialise = True
    elif arg == "-s":
        i += 1
        if i < len(sys.argv):
            numSteps = int(sys.argv[i])
            if numSteps <= 0:
                print("Argument to -s must be positive")
                sys.exit(1)
        else:
            print("No argument for -s", file=sys.stderr)
            sys.exit(1)
    elif arg == "-o":
        i += 1
        if i < len(sys.argv):
            outputImg = sys.argv[i]
        else:
            print("No argument for -o", file=sys.stderr)
            sys.exit(1)
    else:
        if mode == None:
            if arg == "train":
                mode = MODE_TRAIN
            elif arg == "test":
                mode = MODE_TEST
            elif arg == "run":
                mode = MODE_RUN
            elif arg == "samples":
                mode = MODE_SAMPLES
            else:
                print("Unrecognised action", file=sys.stderr)
                sys.exit(1)
        elif dataFile == None:
            dataFile = arg
        elif filterFile == None:
            filterFile = arg
        else:
            print(usage)
            sys.exit(1)
    i += 1
if mode == None:
    print("No specified action", file=sys.stderr)
    sys.exit(1)
if dataFile == None:
    print("No specified data file", file=sys.stderr)
    sys.exit(1)
if filterFile == None:
    filterFile = "filterData.txt"

#constants
IMG_HEIGHT           = 960
IMG_WIDTH            = 1280
IMG_CHANNELS         = 3
IMG_DOWNSCALE        = 2
IMG_SCALED_HEIGHT    = IMG_HEIGHT // IMG_DOWNSCALE
IMG_SCALED_WIDTH     = IMG_WIDTH  // IMG_DOWNSCALE
INPUT_HEIGHT         = 32
INPUT_WIDTH          = 32
INPUT_CHANNELS       = 3
SAVE_FILE            = "modelData/model.ckpt" #save/load network values to/from here
RUN_OUTPUT_IMAGE     = outputImg or "output.jpg"  #output image for 'run' action
SAMPLES_OUTPUT_IMAGE = outputImg or "samples.jpg" #output image for 'sample' action
TRAINING_STEPS       = numSteps
TRAINING_BATCH_SIZE  = 50 #the number of inputs per training step
TRAINING_LOG_PERIOD  = 50 #informative lines are printed after this many training steps
TRAINING_SAVE_PERIOD = 1000 #save after every N steps
TESTING_STEPS        = numSteps #number of batches used for testing
TESTING_BATCH_SIZE   = 50
SUMMARIES_DIR        = 'summaries'

#obtain filter
cellFilter = None #has the form [[flag1, ...], ...], specifying filtered cells
if filterFile != "":
    cellFilter = []
    with open(filterFile) as file:
        for line in file:
            cellFilter.append([int(c) for c in line.strip()])
else:
    cellFilter = [
        [0 for col in IMG_SCALED_WIDTH // INPUT_WIDTH]
        for row in IMG_SCALED_HEIGHT // INPUT_HEIGHT
    ]

#classes for producing input values
class CoarseBatchProducer:
    "Produces input values for the coarse network"
    VALUES_PER_IMAGE = 30
    #constructor
    def __init__(self, dataFile, cellFilter):
        self.filenames = [] #list of image files
        self.cells = []     #has the form [[[c1, c2, ...], ...], ...], specifying cells of images
        self.fileIdx = 0
        self.image = None
        self.valuesGenerated = 0
        self.unfilteredCells = None
        #read 'dataFile' (should have the same format as output by 'markImages.py' with -w)
        cellsDict = dict()
        filename = None
        with open(dataFile) as file:
            for line in file:
                if line[0] != " ":
                    filename = line.strip()
                    self.filenames.append(filename)
                    cellsDict[filename] = []
                else:
                    cellsDict[filename].append([int(c) for c in line.strip()])
        random.shuffle(self.filenames)
        self.cells = [cellsDict[name] for name in self.filenames]
        if len(self.filenames) == 0:
            raise Exception("No filenames")
        #obtain PIL image
        self.image = Image.open(self.filenames[self.fileIdx])
        self.image = self.image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #obtain indices of non-filtered cells (used to randomly select a non-filtered cell)
        rowSize = IMG_SCALED_WIDTH//INPUT_WIDTH
        colSize = IMG_SCALED_HEIGHT//INPUT_HEIGHT
        self.unfilteredCells = []
        for row in range(len(cellFilter)):
            for col in range(len(cellFilter[row])):
                if cellFilter[row][col] == 0:
                    self.unfilteredCells.append(col+row*rowSize)
        if len(self.unfilteredCells) == 0:
            raise Exception("No unfiltered cells")
    #returns a tuple containing a numpy array of 'size' inputs, and a numpy array of 'size' outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        for i in range(size):
            if self.valuesGenerated == self.VALUES_PER_IMAGE:
                #open next image file
                self.fileIdx += 1
                if self.fileIdx+1 > len(self.filenames):
                    self.fileIdx = 0
                self.image = Image.open(self.filenames[self.fileIdx])
                self.image = self.image.resize(
                    (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
                    resample=Image.LANCZOS
                )
                self.valuesGenerated = 0
            #randomly select a non-filtered grid cell
            idx = self.unfilteredCells[
                math.floor(random.random() * len(self.unfilteredCells))
            ]
            rowSize = IMG_SCALED_WIDTH // INPUT_WIDTH
            i = idx % rowSize
            j = idx // rowSize
            x = i*INPUT_WIDTH
            y = j*INPUT_HEIGHT
            #get an input
            cellImg = self.image.crop((x, y, x+INPUT_WIDTH, y+INPUT_HEIGHT))
            cellImg = cellImg.rotate(math.floor(random.random() * 4) * 90) #randomly rotate
            if random.random() > 0.5: #randomly flip
                cellImg = cellImg.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5: #randomly flip
                cellImg = cellImg.transpose(Image.FLIP_TOP_BOTTOM)
            #cellImg = ImageOps.autocontrast(cellImg)
            #cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
            data = np.array(list(cellImg.getdata())).astype(np.float32)
            data = data.reshape((INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS))
            inputs.append(data)
            #get an output
            outputs.append([1, 0] if self.cells[self.fileIdx][j][i] == 1 else [0, 1])
            #update
            self.valuesGenerated += 1
        return np.array(inputs), np.array(outputs).astype(np.float32)
class BatchProducer:
    "Produces input values for the detailed network"
    VALUES_PER_IMAGE = 30
    #constructor
    def __init__(self, dataFile, cellFilter, coarseX, coarseY):
        self.filenames = [] #list of image files
        self.boxes = []     #has the form [[x,y,x2,y2], ...], and specifies boxes for each image file
        self.fileIdx = 0
        self.image = None
        self.valuesGenerated = 0
        self.unfilteredCells = None
        self.coarseX = coarseX
        self.coarseY = coarseY #allows using the coarse network to filter cells
        #read 'dataFile' (should have the same format as output by 'genData.py')
        filenameSet = set()
        boxesDict = dict()
        filename = None
        with open(dataFile) as file:
            for line in file:
                if line[0] != " ":
                    filename = line.strip()
                    filenameSet.add(filename)
                    if not filename in boxesDict:
                        boxesDict[filename] = []
                else:
                    boxesDict[filename].append([int(c) for c in line.strip().split(",")])
        self.filenames = list(filenameSet)
        random.shuffle(self.filenames)
        self.boxes = [boxesDict[name] for name in self.filenames]
        if len(self.filenames) == 0:
            raise Exception("No filenames")
        #obtain PIL image
        self.image = Image.open(self.filenames[self.fileIdx])
        self.image = self.image.resize(
            (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
            resample=Image.LANCZOS
        )
        #obtain indices of non-filtered cells (used to randomly select a non-filtered cell)
        rowSize = IMG_SCALED_WIDTH//INPUT_WIDTH
        colSize = IMG_SCALED_HEIGHT//INPUT_HEIGHT
        self.unfilteredCells = []
        for row in range(len(cellFilter)):
            for col in range(len(cellFilter[row])):
                if cellFilter[row][col] == 0:
                    self.unfilteredCells.append(col+row*rowSize)
        if len(self.unfilteredCells) == 0:
            raise Exception("No unfiltered cells")
    #returns a tuple containing a numpy array of 'size' inputs, and a numpy array of 'size' outputs
    def getBatch(self, size):
        inputs = []
        outputs = []
        potentialInputs = []
        potentialOutputs = []
        while size > 0:
            for i in range(size):
                if self.valuesGenerated == self.VALUES_PER_IMAGE:
                    #open next image file
                    self.fileIdx += 1
                    if self.fileIdx+1 > len(self.filenames):
                        self.fileIdx = 0
                    self.image = Image.open(self.filenames[self.fileIdx])
                    self.image = self.image.resize(
                        (IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
                        resample=Image.LANCZOS
                    )
                    self.valuesGenerated = 0
                #randomly select a non-filtered grid cell
                idx = self.unfilteredCells[
                    math.floor(random.random() * len(self.unfilteredCells))
                ]
                rowSize = IMG_SCALED_WIDTH // INPUT_WIDTH
                i = idx % rowSize
                j = idx // rowSize
                x = i*INPUT_WIDTH
                y = j*INPUT_HEIGHT
                #get an input
                cellImg = self.image.crop((x, y, x+INPUT_WIDTH, y+INPUT_HEIGHT))
                cellImg = cellImg.rotate(math.floor(random.random() * 4) * 90) #randomly rotate
                if random.random() > 0.5: #randomly flip
                    cellImg = cellImg.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() > 0.5: #randomly flip
                    cellImg = cellImg.transpose(Image.FLIP_TOP_BOTTOM)
                #cellImg = ImageOps.autocontrast(cellImg)
                #cellImg = cellImg.filter(ImageFilter.GaussianBlur(1))
                data = np.array(list(cellImg.getdata())).astype(np.float32)
                data = data.reshape((INPUT_WIDTH, INPUT_HEIGHT, IMG_CHANNELS))
                potentialInputs.append(data)
                #get an output
                topLeftX = x*IMG_DOWNSCALE + 15
                topLeftY = y*IMG_DOWNSCALE + 15
                bottomRightX = (x+INPUT_WIDTH-1)*IMG_DOWNSCALE - 15
                bottomRightY = (y+INPUT_HEIGHT-1)*IMG_DOWNSCALE - 15
                hasOverlappingBox = False
                for box in self.boxes[self.fileIdx]:
                    if (not box[2] < topLeftX and
                        not box[0] > bottomRightX and
                        not box[3] < topLeftY and
                        not box[1] > bottomRightY):
                        hasOverlappingBox = True
                        break
                potentialOutputs.append([1, 0] if hasOverlappingBox else [0, 1])
                #update
                self.valuesGenerated += 1
            #filter using coarse network
            out = self.coarseY.eval(feed_dict={self.coarseX: np.array(potentialInputs)})
            unfilteredIndices = [i for i in range(len(potentialInputs)) if out[i][0] < 0.5]
            inputs  += [potentialInputs[i] for i in unfilteredIndices]
            outputs += [potentialOutputs[i] for i in unfilteredIndices]
            #update
            size -= len(unfilteredIndices)
            potentialInputs = []
            potentialOutputs = []
        return np.array(inputs), np.array(outputs).astype(np.float32)

#create computation graph
with tf.name_scope('input'): #group nodes for easier viewing with tensorboard
    x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_input')
    p_dropout = tf.placeholder(tf.float32, name='p_dropout')
def createCoarseNetwork(x, y_):
    NET_NAME = 'coarse_net'
    summaries = []
    with tf.name_scope(NET_NAME):
        with tf.name_scope('input_reshape'):
            x_flat = tf.reshape(x, [-1, INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS])
            #add summary
            summaries.append(tf.image_summary(NET_NAME + '/input', x, 10))
        with tf.name_scope('output'):
            with tf.name_scope('weights'):
                w = tf.Variable(
                    tf.truncated_normal([INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS, 2], stddev=0.1)
                )
                #add summaries
                mean = tf.reduce_mean(w)
                summaries.append(tf.scalar_summary(NET_NAME + '/mean/weights', mean))
                summaries.append(tf.scalar_summary(
                    NET_NAME + '/stddev/weights', tf.reduce_mean(tf.square(w - mean))
                ))
                summaries.append(tf.histogram_summary(NET_NAME + '/weights', w))
            with tf.name_scope('biases'):
                b = tf.Variable(tf.constant(0.1, shape=[2]))
                #add summaries
                mean = tf.reduce_mean(b)
                summaries.append(tf.scalar_summary(NET_NAME + '/mean/biases', mean))
                summaries.append(tf.scalar_summary(NET_NAME + '/stddev/biases', tf.reduce_mean(tf.square(b - mean))))
                summaries.append(tf.histogram_summary(NET_NAME + '/biases', b))
            y = tf.nn.sigmoid(tf.matmul(x_flat, w) + b, 'y-output')
        #cost
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.square(y_ - y), reduction_indices=[1])
            #add summary
            summaries.append(tf.scalar_summary(NET_NAME + '/cost', tf.reduce_mean(cost)))
        #optimizer
        with tf.name_scope('train'):
            train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
        #accuracy
        with tf.name_scope('accuracy'):
            correctness = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
            #add summary
            summaries.append(tf.scalar_summary(NET_NAME + '/accuracy', accuracy))
    #variables
    variables = [w, b]
    #return output nodes and trainer
    return y, accuracy, train, variables, tf.merge_summary(summaries)
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
            correctness = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
            #add summary
            summaries.append(tf.scalar_summary(NET_NAME + '/accuracy', accuracy))
    #variables
    variables = [w1, b1, w2, b2, w3, b3, w4, b4]
    #return output nodes and trainer
    return y, accuracy, train, variables, tf.merge_summary(summaries)
cy, caccuracy, ctrain, cvariables, csummaries = createCoarseNetwork(x, y_)
y, accuracy, train, variables, summaries = createDetailedNetwork(x, y_, p_dropout)

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
    if mode == MODE_TRAIN:
        if useCoarseOnly:
            #train coarse network
            summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/train/coarse', sess.graph)
            prod = CoarseBatchProducer(dataFile, cellFilter)
            startTime = time.time()
            for step in range(TRAINING_STEPS):
                inputs, outputs = prod.getBatch(TRAINING_BATCH_SIZE)
                summary, _ = sess.run(
                    [csummaries, ctrain],
                    feed_dict={x: inputs, y_: outputs}
                )
                summaryWriter.add_summary(summary, step)
                if step % TRAINING_LOG_PERIOD == 0 or step == TRAINING_STEPS-1:
                    acc = caccuracy.eval(feed_dict={x: inputs, y_: outputs})
                    print("%7.2f secs - step %d, accuracy %g" % (time.time() - startTime, step, acc))
                if step > 0 and step % TRAINING_SAVE_PERIOD == 0:
                    saver.save(sess, SAVE_FILE)
            summaryWriter.close()
        else:
            #train detailed network
            summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/train/detailed', sess.graph)
            prod = BatchProducer(dataFile, cellFilter, x, cy)
            startTime = time.time()
            for step in range(TRAINING_STEPS):
                inputs, outputs = prod.getBatch(TRAINING_BATCH_SIZE)
                summary, _ = sess.run(
                    [summaries, train],
                    feed_dict={x: inputs, y_: outputs, p_dropout: 0.5}
                )
                summaryWriter.add_summary(summary, step)
                if step % TRAINING_LOG_PERIOD == 0 or step == TRAINING_STEPS-1:
                    acc = accuracy.eval(feed_dict={x: inputs, y_: outputs, p_dropout: 1.0})
                    print("%7.2f secs - step %d, accuracy %g" % (time.time()-startTime, step, acc))
                if step > 0 and step % TRAINING_SAVE_PERIOD == 0:
                    saver.save(sess, SAVE_FILE)
            summaryWriter.close()
    #testing
    elif mode == MODE_TEST:
        if useCoarseOnly:
            #test coarse network
            summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/test/coarse', sess.graph)
            prod = CoarseBatchProducer(dataFile, cellFilter)
            startTime = time.time()
            accuracies = []
            for step in range(TESTING_STEPS):
                inputs, outputs = prod.getBatch(TESTING_BATCH_SIZE)
                summary, acc = sess.run(
                    [csummaries, caccuracy],
                    feed_dict={x: inputs, y_: outputs}
                )
                summaryWriter.add_summary(summary, step)
                accuracies.append(acc)
                print("%7.2f secs - step %d, accuracy %g" % (time.time()-startTime, step, acc))
            avgAcc = sum(accuracies)/len(accuracies)
            print("average accuracy %g" % avgAcc)
            summaryWriter.close()
        else:
            #test detailed network
            summaryWriter = tf.train.SummaryWriter(SUMMARIES_DIR + '/test/detailed', sess.graph)
            prod = BatchProducer(dataFile, cellFilter, x, cy)
            startTime = time.time()
            accuracies = []
            for step in range(TESTING_STEPS):
                inputs, outputs = prod.getBatch(TESTING_BATCH_SIZE)
                summary, acc = sess.run(
                    [summaries, accuracy],
                    feed_dict={x: inputs, y_: outputs, p_dropout: 1.0}
                )
                summaryWriter.add_summary(summary, step)
                accuracies.append(acc)
                print("%7.2f secs - step %d, accuracy %g" % (time.time()-startTime, step, acc))
            avgAcc = sum(accuracies)/len(accuracies)
            print("average accuracy %g" % avgAcc)
            summaryWriter.close()
    #running
    elif mode == MODE_RUN:
        #obtain PIL image
        image = Image.open(dataFile)
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
        if useCoarseOnly:
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
                        p[i][j] = out[0][0]
        else:
            for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
                for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                    if cellFilter != None and cellFilter[i][j] == 1:
                        p[i][j] = -2
                    else:
                        coarseOut = cy.eval(feed_dict={
                            x: array[:, INPUT_HEIGHT*i:INPUT_HEIGHT*(i+1), INPUT_WIDTH*j:INPUT_WIDTH*(j+1), :]
                        })
                        if coarseOut[0][0] > 0.5:
                            p[i][j] = -1
                        else:
                            out = y.eval(feed_dict={
                                x: array[:, INPUT_HEIGHT*i:INPUT_HEIGHT*(i+1), INPUT_WIDTH*j:INPUT_WIDTH*(j+1), :],
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
        image.save(RUN_OUTPUT_IMAGE)
        #output time taken
        print("%7.2f secs - output written to %s" % (time.time() - startTime, RUN_OUTPUT_IMAGE))
    #sample generating
    elif mode == MODE_SAMPLES:
        NUM_SAMPLES = (20, 20)
        if useCoarseOnly:
            prod = CoarseBatchProducer(dataFile, cellFilter)
        else:
            prod = BatchProducer(dataFile, cellFilter, x, cy)
        image = Image.new("RGB", (INPUT_WIDTH*NUM_SAMPLES[0], INPUT_HEIGHT*NUM_SAMPLES[1]))
        draw = ImageDraw.Draw(image, "RGBA")
        #get samples
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
                if outputs[0][0] > 0.5:
                    draw.rectangle([
                        INPUT_WIDTH*i,
                        INPUT_HEIGHT*j,
                        INPUT_WIDTH*(i+1),
                        INPUT_HEIGHT*(j+1),
                    ], fill=(0,255,0,64))
        image.save(SAMPLES_OUTPUT_IMAGE)
        #output time taken
        print("%7.2f secs - output written to %s" % (time.time() - startTime, SAMPLES_OUTPUT_IMAGE))
    #saving
    saver.save(sess, SAVE_FILE)
