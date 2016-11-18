import math, os, time
import tensorflow as tf

from .constants import *

class Network:
    """ Holds nodes for a tensorflow network """
    def __init__(self, graph, x, y_, p_dropout, y, accuracy, precision, recall, train, summaries):
        self.graph = graph
        self.x = x
        self.y_ = y_
        self.p_dropout = p_dropout
        self.y = y
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.train = train
        self.summaries = tf.merge_summary(summaries)

def createCoarseNetwork(graph, threshold):
    #parameters for network variations
    PREPROCESS_NORMALIZE = True
    ACTIVATION_FUNC = tf.nn.sigmoid #tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, prelu
    OUTPUT_ACTIVATION_FUNC = tf.nn.sigmoid #tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, prelu
    HIDDEN_LAYERS = [30, 10]
    COST_FUNC = "squared_error" #"squared_error", "logistic_loss", "softmax_cross_entropy_with_logits"
    OPTIMIZER = "adam" #"adam", "gradient_descent", "adadelta", "adagrad", "momentum"
    USE_DROPOUT = False
    #helper functions
    def createLayer(input, inSize, outSize, layerName, summaries, activation=ACTIVATION_FUNC):
        with tf.name_scope(layerName):
            with tf.name_scope("weights"):
                w = tf.Variable(tf.truncated_normal([inSize, outSize]))
                addSummaries(w, summaries, layerName + "/weights", "mean_stddev_hist")
            with tf.name_scope("biases"):
                b = tf.Variable(tf.constant(1.0, shape=[outSize]))
                addSummaries(b, summaries, layerName + "/biases", "mean_stddev_hist")
            wb = tf.matmul(input, w) + b
            if USE_DROPOUT:
                wb = tf.nn.dropout(wb, p_dropout)
            return activation(wb, name="out")
    #create nodes
    summaries = []
    with graph.as_default():
        with tf.name_scope("coarse_net"):
            #input nodes
            with tf.name_scope("input"):
                x = tf.placeholder(tf.float32, \
                    [None, INPUT_HEIGHT, INPUT_WIDTH, IMG_CHANNELS], name="x_input")
                y_ = tf.placeholder(tf.float32, [None, 2], name="y_input")
                p_dropout = tf.placeholder(tf.float32, name="p_dropout")
            with tf.name_scope("process_input"):
                if PREPROCESS_NORMALIZE:
                    x2 = tf.div(x, tf.constant(255.0))
                else:
                    x2 = x
                x_flat = tf.reshape(x2, [-1, INPUT_HEIGHT*INPUT_WIDTH*IMG_CHANNELS])
                addSummaries(x2, summaries, "input", "image")
            #hidden and output layers
            layerSizes = [INPUT_HEIGHT*INPUT_WIDTH*IMG_CHANNELS] + HIDDEN_LAYERS
            layer = x_flat
            for i in range(1,len(layerSizes)):
                layer = createLayer(
                    layer, layerSizes[i-1], layerSizes[i], "hidden_layer" + str(i), summaries
                )
            y = createLayer(
                layer, layerSizes[-1], 2, "output_layer", summaries, OUTPUT_ACTIVATION_FUNC
            )
            #cost
            with tf.name_scope("cost"):
                if COST_FUNC == "squared_error":
                    cost = tf.square(y_ - y)
                elif COST_FUNC == "logistic_loss":
                    cost = tf.constant(1/math.log(2)) * tf.log(tf.constant(1.0) + tf.exp(-y * y_))
                elif COST_FUNC == "softmax_cross_entropy_with_logits":
                    cost = tf.nn.softmax_cross_entropy_with_logits(y, y_)
                else:
                    raise Exception("Unrecognised cost function")
                addSummaries(cost, summaries, "cost", "mean")
            #optimizer
            with tf.name_scope("train"):
                if OPTIMIZER == "adam":
                    train = tf.train.AdamOptimizer().minimize(cost)
                elif OPTIMIZER == "gradient_descent":
                    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
                elif OPTIMIZER == "adadelta":
                    train = tf.train.AdadeltaOptimizer().minimize(cost)
                elif OPTIMIZER == "adagrad":
                    train = tf.train.AdagradOptimizer(0.01).minimize(cost)
                elif OPTIMIZER == "momentum":
                    train = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cost)
                elif OPTIMIZER == "ftrl":
                    train = tf.train.FtrlOptimizer(0.01).minimize(cost)
                elif OPTIMIZER == "rmsprop":
                    train = tf.train.RMSPropOptimizer(0.01).minimize(cost)
                else:
                    raise Exception("Unrecognised optimizer")
            #metrics
            with tf.name_scope("metrics"):
                y_pred = tf.greater(tf.slice(y, [0, 0], [-1, 1]), tf.constant(threshold))
                y2_pred = tf.greater(tf.slice(y_, [0, 0], [-1, 1]), tf.constant(0.5))
                correctness = tf.equal(y_pred, y2_pred)
                #accuracy
                accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
                addSummaries(accuracy, summaries, "accuracy", "mean")
                #precision and recall
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
                addSummaries(prec, summaries, "precision", "mean")
                addSummaries(rec, summaries, "recall", "mean")
    #return output nodes and trainer
    return Network(graph, x, y_, p_dropout, y, accuracy, prec, rec, train, summaries)

def createDetailedNetwork(graph):
    #parameters for network variations
    PREPROCESS_NORMALIZE = True
    CONV_LAYERS = [16, 32]
    CONV_STRIDE = 1
    POOL_STRIDE = 2
    DENSE_LAYERS = [64]
    ACTIVATION_FUNC = tf.nn.relu #tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, prelu
    #helper functions
    def createWeights(shape):
        with tf.name_scope("weights"):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    def createBiases(shape):
        with tf.name_scope("biases"):
            return tf.Variable(tf.constant(1.0, shape=shape))
    def createConv(x, w, b):
        with tf.name_scope("conv"):
            xw = tf.nn.conv2d(x, w, strides=[1, CONV_STRIDE, CONV_STRIDE, 1], padding="SAME")
            return tf.nn.relu(xw + b)
    def createPool(c):
        with tf.name_scope("pool"):
            return tf.nn.max_pool(c, ksize=[1, POOL_STRIDE, POOL_STRIDE, 1], \
                strides=[1, POOL_STRIDE, POOL_STRIDE, 1], padding="SAME")
    #create nodes
    summaries = []
    with graph.as_default():
        with tf.name_scope("detailed_net"):
            #input nodes
            with tf.name_scope("input"): #group nodes for easier viewing with tensorboard
                x = tf.placeholder(tf.float32, \
                    [None, INPUT_HEIGHT, INPUT_WIDTH, IMG_CHANNELS], name="x_input")
                y_ = tf.placeholder(tf.float32, [None, 2], name="y_input")
                p_dropout = tf.placeholder(tf.float32, name="p_dropout")
            with tf.name_scope("process_input"):
                if PREPROCESS_NORMALIZE:
                    x2 = tf.div(x, tf.constant(255.0))
                else:
                    x2 = x
                addSummaries(x2, summaries, "input", "image")
            #convolutional layers
            convSizes = [IMG_CHANNELS] + CONV_LAYERS
            input = x2
            for i in range(1, len(convSizes)):
                with tf.name_scope("conv_layer" + str(i)):
                    w = createWeights([5, 5, convSizes[i-1], convSizes[i]])
                        #filter_height, filter_width, in_channels, out_channels
                    b = createBiases([convSizes[i]])
                    c = createConv(input, w, b)
                    input = createPool(c)
            #flatten input
            numInputs = INPUT_HEIGHT // (POOL_STRIDE * len(CONV_LAYERS)) * \
                INPUT_WIDTH // (POOL_STRIDE * len(CONV_LAYERS)) * convSizes[-1]
            input = tf.reshape(input, [-1, numInputs])
            #densely connected layers
            denseSizes = [numInputs] + DENSE_LAYERS
            for i in range(1, len(denseSizes)):
                with tf.name_scope("dense_layer" + str(i)):
                    w = createWeights([denseSizes[i-1], denseSizes[i]])
                    b = createBiases([denseSizes[i]])
                    input = ACTIVATION_FUNC(tf.matmul(input, w) + b)
                    input = tf.nn.dropout(input, p_dropout)
            #readout layer
            with tf.name_scope("readout_layer"):
                w = createWeights([denseSizes[-1], 2])
                b = createBiases([2])
                y  = tf.nn.softmax(tf.matmul(input, w) + b)
            #cost
            with tf.name_scope("cost"):
                cost = -(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
                addSummaries(cost, summaries, "cost", "mean")
            #optimizer
            with tf.name_scope("train"):
                train = tf.train.AdamOptimizer().minimize(cost)
            #metrics
            with tf.name_scope("metrics"):
                y_pred  = tf.greater(tf.slice(y,  [0, 0], [-1, 1]), tf.slice(y,  [0, 1], [-1, 1]))
                y2_pred = tf.greater(tf.slice(y_, [0, 0], [-1, 1]), tf.slice(y_, [0, 1], [-1, 1]))
                correctness = tf.equal(y_pred, y2_pred)
                #accuracy
                accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
                addSummaries(accuracy, summaries, "accuracy", "mean")
                #precision and recall
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
                addSummaries(prec, summaries, "precision", "mean")
                addSummaries(rec, summaries, "recall", "mean")
    #return output nodes and trainer
    return Network(graph, x, y_, p_dropout, y, accuracy, prec, rec, train, summaries)

def addSummaries(node, summaries, name, method):
    """ Used to create and attach summary nodes to "node".
        "method" specifies the kinds of summaries to add.
    """
    with tf.device("/cpu:0"): #it seems summary nodes don't work for GPUs
        if method == "mean":
            summaries.append(tf.scalar_summary(name + "/mean", tf.reduce_mean(node)))
        elif method == "mean_stddev_hist":
            mean = tf.reduce_mean(node)
            summaries.append(tf.scalar_summary(name + "/mean", mean))
            summaries.append(tf.scalar_summary(name + "/stddev", tf.reduce_mean(tf.square(node-mean))))
            summaries.append(tf.histogram_summary(name, node))
        elif method == "image":
            MAX_NUM_IMAGES = 10
            summaries.append(tf.image_summary(name, node, MAX_NUM_IMAGES))

def prelu(x, name=None):
    alphas = tf.Variable(tf.constant(0.0, shape=[x.get_shape()[-1]]))
    return tf.add(tf.nn.relu(x), alphas * (x - abs(x)) * 0.5)
