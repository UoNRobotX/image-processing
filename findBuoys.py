import sys, os, math, random
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

usage = "Usage: python3 " + sys.argv[0] + """ [-cdn] [-f df1] [-t df1] [-e df1] [-r img1] [-s df1]
    Loads/trains/tests/runs the coarse/detailed networks.
    By default, network values are loaded from files if they exist, and nothing is done.
    If neither -c nor -d is given, the default is -d.

    Options:
        -c
            With -n, only re-initialise values for the coarse network.
            With -t, train the coarse network.
            With -e, test the coarse network, ignoring the detailed network.
            With -r, run the coarse network on the image, ignoring the detailed network.
            With -s, generate inputs to the coarse network.
        -d
            With -n, only re-initialise values for the detailed network.
            With -t, train the detailed network, using the coarse network to filter input.
            With -e, test the detailed network, using the coarse network to filter input.
            With -r, run the detailed network on the image, using the coarse network to filter input.
            With -s, generate inputs to the detailed network, using the coarse network to filter input.
        -n
            Re-initialise values for the coarse/detailed network.
        -f df1
            Ignore grid cells specified by a filter from file 'df1'.
        -t df1
            Train the coarse/detailed network, using training data from file 'df1'.
        -e df1
            Test coarse/detailed networks, using testing data from file 'df1'.
        -r img1
            Run coarse/detailed networks using input image 'img1'.
        -s df1
            Generate input samples for the coarse/detailed network, using data from file 'df1'.
"""

#process command line arguments
useCoarse    = False
filterFile   = None
trainingFile = None
testingFile  = None
runningFile  = None
samplesFile  = None
reinitialise = False
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "-c":
        useCoarse = True
    elif arg == "-d":
        useCoarse = False
    elif arg == "-f":
        i += 1
        if i < len(sys.argv):
            filterFile = sys.argv[i]
        else:
            print("No argument for -f", file=sys.stderr)
            sys.exit(1)
    elif arg == "-t":
        i += 1
        if i < len(sys.argv):
            trainingFile = sys.argv[i]
        else:
            print("No argument for -t", file=sys.stderr)
            sys.exit(1)
    elif arg == "-e":
        i += 1
        if i < len(sys.argv):
            testingFile = sys.argv[i]
        else:
            print("No argument for -e", file=sys.stderr)
            sys.exit(1)
    elif arg == "-r":
        i += 1
        if i < len(sys.argv):
            runningFile = sys.argv[i]
        else:
            print("No argument for -r", file=sys.stderr)
            sys.exit(1)
    elif arg == "-s":
        i += 1
        if i < len(sys.argv):
            samplesFile = sys.argv[i]
        else:
            print("No argument for -s", file=sys.stderr)
            sys.exit(1)
    elif arg == "-n":
        reinitialise = True
    else:
        print(usage)
        sys.exit(0)
    i += 1

#read 'filterFile' if given
cellFilter = None #has the form [[flag1, ...], ...], specifying filtered cells
if filterFile != None:
    cellFilter = []
    with open(filterFile) as file:
        for line in file:
            cellFilter.append([int(c) for c in line.strip()])

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
SAVE_FILE            = "modelData/model.ckpt"   #save/load network values to/from here
RUN_OUTPUT_IMAGE     = "outputFindBuoys.jpg"  #with -r, a representation of the output is saved here
SAMPLES_OUTPUT_IMAGE = "samplesFindBuoys.jpg" #with -s, a representation of the output is saved here
TRAINING_STEPS       = 100 #with -t, the number of training iterations
TRAINING_BATCH_SIZE  = 50  #with -t, the number of inputs per training iteration
TRAINING_LOG_PERIOD  = 50  #with -t, informative lines are printed after this many training iterations
TESTING_BATCH_SIZE   = 50  #with -e, the number of inputs used for testing

#classes for producing input values
class CoarseBatchProducer:
    "Produces input values for the coarse network"
    VALUES_PER_IMAGE = 30
    #constructor
    def __init__(self, dataFile, cellFilter):
        self.filenames = [] #list of image files
        self.cells = []     #has the form [[[c1, c2, ...], ...], ...], specifying cells of image files
        self.fileIdx = 0
        self.image = None
        self.data = None
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
        #obtain numpy array
        self.data = np.array(list(self.image.getdata())).astype(np.float32)
        self.data = self.data.reshape((IMG_SCALED_HEIGHT, IMG_SCALED_WIDTH, IMG_CHANNELS))
        #obtain indices of non-filtered cells (used to randomly select a non-filtered cell)
        rowSize = IMG_SCALED_WIDTH//INPUT_WIDTH
        colSize = IMG_SCALED_HEIGHT//INPUT_HEIGHT
        if cellFilter != None:
            self.unfilteredCells = []
            for row in range(len(cellFilter)):
                for col in range(len(cellFilter[row])):
                    if cellFilter[row][col] == 0:
                        self.unfilteredCells.append(col+row*rowSize)
            if len(self.unfilteredCells) == 0:
                raise Exception("No unfiltered cells")
        else:
            self.unfilteredCells = range(colSize * rowSize)
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
                self.data = np.array(list(self.image.getdata())).astype(np.float32)
                self.data = self.data.reshape((IMG_SCALED_HEIGHT, IMG_SCALED_WIDTH, IMG_CHANNELS))
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
            inputs.append(self.data[y:y+INPUT_HEIGHT, x:x+INPUT_WIDTH, :])
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
        self.data = None
        self.valuesGenerated = 0
        self.unfilteredCells = None
        self.coarseX = coarseX
        self.coarseY = coarseY #allows using the coarse network to filter cells
        #read 'dataFile' (should have the same format as output by 'genData.py')
        filenameSet = set()
        boxesDict = dict()
        with open(dataFile) as file:
            for line in file:
                record = line.strip().split(",")
                filenameSet.add(record[0])
                if not record[0] in boxesDict:
                    boxesDict[record[0]] = [[int(field) for field in record[1:5]]]
                else:
                    boxesDict[record[0]].append([int(field) for field in record[1:5]])
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
        #obtain numpy array
        self.data = np.array(list(self.image.getdata())).astype(np.float32)
        self.data = self.data.reshape((IMG_SCALED_HEIGHT, IMG_SCALED_WIDTH, IMG_CHANNELS))
        #obtain indices of non-filtered cells (used to randomly select a non-filtered cell)
        rowSize = IMG_SCALED_WIDTH//INPUT_WIDTH
        colSize = IMG_SCALED_HEIGHT//INPUT_HEIGHT
        if cellFilter != None:
            self.unfilteredCells = []
            for row in range(len(cellFilter)):
                for col in range(len(cellFilter[row])):
                    if cellFilter[row][col] == 0:
                        self.unfilteredCells.append(col+row*rowSize)
            if len(self.unfilteredCells) == 0:
                raise Exception("No unfiltered cells")
        else:
            self.unfilteredCells = range(colSize * rowSize)
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
                    self.data = np.array(list(self.image.getdata())).astype(np.float32)
                    self.data = self.data.reshape((IMG_SCALED_HEIGHT, IMG_SCALED_WIDTH, IMG_CHANNELS))
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
                potentialInputs.append(self.data[y:y+INPUT_HEIGHT, x:x+INPUT_WIDTH, :])
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
            print(size)
            potentialInputs = []
            potentialOutputs = []
        return np.array(inputs), np.array(outputs).astype(np.float32)

#create computation graph
x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS])
y_ = tf.placeholder(tf.float32, [None, 2])
p_dropout = tf.placeholder(tf.float32)
def createCoarseNetwork(x, y_):
    x_flat = tf.reshape(x, [-1, INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS])
    w = tf.Variable(tf.truncated_normal([INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS, 2], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[2]))
    y = tf.nn.sigmoid(tf.matmul(x_flat, w) + b)
    #cost
    cost = tf.reduce_mean(tf.square(y_ - y), reduction_indices=[1])
    #optimizer
    train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    #accuracy
    correctness = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
    #variables
    variables = [w, b]
    #return output nodes and trainer
    return y, accuracy, train, variables
def createDetailedNetwork(x, y_, p_dropout):
    #helper functions
    def createWeights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    def createBiases(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))
    def createConv(x, w, b):
        xw = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        return tf.nn.relu(xw + b)
    def createPool(c):
        return tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    #first convolutional layer
    w1 = createWeights([5, 5, 3, 32]) #filter_height, filter_width, in_channels, out_channels
    b1 = createBiases([32])
    c1 = createConv(x, w1, b1)
    p1 = createPool(c1)
    #second convolutional layer
    w2 = createWeights([5, 5, 32, 64])
    b2 = createBiases([64])
    c2 = createConv(p1, w2, b2)
    p2 = createPool(c2)
    #densely connected layer
    w3 = createWeights([INPUT_HEIGHT//4 * INPUT_WIDTH//4 * 64, 1024])
    b3 = createBiases([1024])
    p2_flat = tf.reshape(p2, [-1, INPUT_HEIGHT//4 * INPUT_WIDTH//4 * 64])
    h1 = tf.nn.relu(tf.matmul(p2_flat, w3) + b3)
    #dropout
    h1_dropout = tf.nn.dropout(h1, p_dropout)
    #readout layer
    w4 = createWeights([1024, 2])
    b4 = createBiases([2])
    y  = tf.nn.softmax(tf.matmul(h1_dropout, w4) + b4)
    #cost
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)),
        reduction_indices=[1])
    )
    #optimizer
    train = tf.train.AdamOptimizer().minimize(cross_entropy)
    #accuracy
    correctness = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
    #variables
    variables = [w1, b1, w2, b2, w3, b3, w4, b4]
    #return output nodes and trainer
    return y, accuracy, train, variables
cy, caccuracy, ctrain, cvariables = createCoarseNetwork(x, y_)
y, accuracy, train, variables = createDetailedNetwork(x, y_, p_dropout)

#create savers
saver = tf.train.Saver(tf.all_variables())

#use graph
with tf.Session() as sess:
    #initialising
    if os.path.exists(SAVE_FILE):
        saver.restore(sess, SAVE_FILE)
    if reinitialise:
        if useCoarse:
            sess.run(tf.initialize_variables(cvariables))
        else:
            sess.run(tf.initialize_variables(variables))
    #training
    if (trainingFile != None):
        if useCoarse:
            #train coarse network
            prod = CoarseBatchProducer(trainingFile, cellFilter)
            for step in range(TRAINING_STEPS):
                inputs, outputs = prod.getBatch(TRAINING_BATCH_SIZE)
                ctrain.run(feed_dict={x: inputs, y_: outputs})
                if step % TRAINING_LOG_PERIOD == 0:
                    acc = caccuracy.eval(feed_dict={x: inputs, y_: outputs})
                    print("step %d, accuracy %g" % (step, acc))
        else:
            #train detailed network
            prod = BatchProducer(trainingFile, cellFilter, x, cy)
            for step in range(TRAINING_STEPS):
                inputs, outputs = prod.getBatch(TRAINING_BATCH_SIZE)
                train.run(feed_dict={x: inputs, y_: outputs, p_dropout: 0.5})
                if step % TRAINING_LOG_PERIOD == 0:
                    acc = accuracy.eval(feed_dict={x: inputs, y_: outputs, p_dropout: 1.0})
                    print("step %d, accuracy %g" % (step, acc))
    #testing
    if (testingFile != None):
        if useCoarse:
            #test coarse network
            prod = CoarseBatchProducer(testingFile, cellFilter)
            inputs, outputs = prod.getBatch(TESTING_BATCH_SIZE)
            acc = caccuracy.eval(feed_dict={x: inputs, y_: outputs})
            print("test accuracy %g" % acc)
        else:
            #test detailed network
            prod = BatchProducer(testingFile, cellFilter, x, cy)
            inputs, outputs = prod.getBatch(TESTING_BATCH_SIZE)
            acc = accuracy.eval(feed_dict={x: inputs, y_: outputs, p_dropout: 1.0})
            print("test accuracy: %g" % acc)
    #running on an image file
    if (runningFile != None):
        #obtain PIL image
        image = Image.open(runningFile)
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
        if useCoarse:
            for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
                for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
                    if cellFilter != None and cellFilter[i][j] == 1:
                        p[i][j] = -1
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
    #generating input samples
    if (samplesFile != None):
        NUM_SAMPLES = (20, 20)
        if useCoarse:
            prod = CoarseBatchProducer(samplesFile, cellFilter)
        else:
            prod = BatchProducer(samplesFile, cellFilter, x, cy)
        image = Image.new("RGB", (INPUT_WIDTH*NUM_SAMPLES[0], INPUT_HEIGHT*NUM_SAMPLES[1]))
        draw = ImageDraw.Draw(image, "RGBA")
        #get samples
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
    #saving
    saver.save(sess, SAVE_FILE)
