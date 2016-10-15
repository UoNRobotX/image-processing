import sys, os, math, random
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tensorflow as tf

usage = "Usage: python3 " + sys.argv[0] + """ [-t trData1] [-e testData1] [-r image1] [-s data1] [-n]
    Creates/trains/tests/runs a neural network for recognising buoys.
    By default, saved network values are loaded, if the corresponding files exist.

    Options:
        -t trData1
            Train the network.
            'trData1' specifies images and bounding boxes.
        -e testData1
            Test the network.
            'testData1' specifies images and bounding boxes.
        -r image1
            Run the network on an input image, specified by 'image1'.
        -s
            Generate input samples.
            'testData1' specifies images and bounding boxes.
        -n
            Do not load a saved network.
"""

#process command line arguments
trainingFile = None
testingFile  = None
runningFile  = None
samplesFile  = None
loadSaved    = True
i = 1
while i < len(sys.argv):
	arg = sys.argv[i]
	if arg == "-t":
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
		loadSaved = False
	else:
		print(usage)
		sys.exit(0)
	i += 1

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
SAVE_FILE            = "modelData/model.ckpt"
RUN_OUTPUT_IMAGE     = "outputFindBuoys.jpg"
SAMPLES_OUTPUT_IMAGE = "samplesFindBuoys.jpg"
TRAINING_STEPS       = 200
TRAINING_BATCH_SIZE  = 50
TRAINING_LOG_PERIOD  = 100
TESTING_BATCH_SIZE   = 50

#class for producing input data
class BatchProducer:
	"Produces batches of training/test data"
	VALUES_PER_IMAGE = 100
	def __init__(self, dataFile):
		self.filenames = []
		self.boxes = []
		self.fileIdx = 0
		self.image = None
		self.valuesGenerated = 0
		#read data file
		filenameSet = set()
		boxesDict = dict()
		with open(dataFile) as file:
			for line in file:
				record = line.strip().split(",")
				filenameSet.add(record[0])
				if record[0] in boxesDict:
					boxesDict[record[0]].append([int(field) for field in record[1:5]])
				else:
					boxesDict[record[0]] = [[int(field) for field in record[1:5]]]
		self.filenames = list(filenameSet)
		random.shuffle(self.filenames)
		self.boxes = [boxesDict[name] for name in self.filenames]
		if len(self.filenames) == 0:
			raise Exception("no filenames")
		#obtain PIL image
		self.image = Image.open(self.filenames[self.fileIdx])
		self.image = self.image.resize(
			(IMG_SCALED_WIDTH, IMG_SCALED_HEIGHT),
			resample=Image.LANCZOS
		)
		#obtain numpy array
		self.data = np.array(list(self.image.getdata())).astype(np.float32)
		self.data = self.data.reshape((IMG_SCALED_HEIGHT, IMG_SCALED_WIDTH, IMG_CHANNELS))
	def getBatch(self, size):
		inputs = []
		outputs = []
		while size > 0:
			if self.valuesGenerated < self.VALUES_PER_IMAGE:
				#get an input
				x = math.floor(random.random()*(IMG_SCALED_WIDTH  - INPUT_WIDTH))
				y = math.floor(random.random()*(IMG_SCALED_HEIGHT - INPUT_HEIGHT))
				inputs.append(self.data[y:y+INPUT_HEIGHT, x:x+INPUT_WIDTH, :])
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
				outputs.append([1, 0] if hasOverlappingBox else [0, 1])
				#update
				self.valuesGenerated += 1
			else:
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
				continue
			size -= 1
		return np.array(inputs), np.array(outputs).astype(np.float32)

#create computation graph
x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS])
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
p_dropout = tf.placeholder(tf.float32)
h1_dropout = tf.nn.dropout(h1, p_dropout)
#readout layer
w4 = createWeights([1024, 2])
b4 = createBiases([2])
y  = tf.nn.softmax(tf.matmul(h1_dropout, w4) + b4)
#cost
y2 = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(
	-tf.reduce_sum(y2 * tf.log(tf.clip_by_value(y,1e-10,1.0)),
	reduction_indices=[1])
)
#optimizer
train = tf.train.AdamOptimizer().minimize(cross_entropy)
#accuracy
correctness = tf.equal(tf.argmax(y, 1), tf.argmax(y2, 1))
accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

#create saver
saver = tf.train.Saver(tf.all_variables())

#use graph
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	#loading
	if loadSaved:
		if os.path.exists(SAVE_FILE):
			saver.restore(sess, SAVE_FILE)
		else:
			print("Save file does not exist", file=sys.stderr)
	#training
	if (trainingFile != None):
		prod = BatchProducer(trainingFile)
		for step in range(TRAINING_STEPS):
			inputs, outputs = prod.getBatch(TRAINING_BATCH_SIZE)
			train.run(feed_dict={x: inputs, y2: outputs, p_dropout: 0.5})
			if step % TRAINING_LOG_PERIOD == 0:
				acc = accuracy.eval(feed_dict={x: inputs, y2: outputs, p_dropout: 1.0})
				print("step %d, accuracy %g" % (step, acc))
	#testing
	if (testingFile != None):
		prod = BatchProducer(testingFile)
		inputs, outputs = prod.getBatch(TESTING_BATCH_SIZE)
		acc = accuracy.eval(feed_dict={x: inputs, y2: outputs, p_dropout: 1.0})
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
		for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
			for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
				output = y.eval(feed_dict={
					x: array[:, INPUT_HEIGHT*i:INPUT_HEIGHT*(i+1), INPUT_WIDTH*j:INPUT_WIDTH*(j+1), :],
					p_dropout: 1.0
				})
				p[i][j] = output[0][0]
		#write results to image file
		draw = ImageDraw.Draw(image, "RGBA")
		for i in range(IMG_SCALED_HEIGHT//INPUT_HEIGHT):
			for j in range(IMG_SCALED_WIDTH//INPUT_WIDTH):
				draw.rectangle([
					INPUT_WIDTH*IMG_DOWNSCALE*j,
					INPUT_HEIGHT*IMG_DOWNSCALE*i + int(INPUT_HEIGHT*IMG_DOWNSCALE*(1-p[i][j])),
					INPUT_WIDTH*IMG_DOWNSCALE*(j+1),
					INPUT_HEIGHT*IMG_DOWNSCALE*(i+1),
				], fill=(0,255,0,64))
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
		prod = BatchProducer(samplesFile)
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
