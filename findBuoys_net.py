import tensorflow as tf

from constants import *

class NetworkNodes:
    """ Holds nodes of a tensorflow network """
    def __init__(self, y, accuracy, precision, recall, train, variables, summaries):
        self.y = y
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.train = train
        self.variables = variables
        self.summaries = summaries

def createCoarseNetwork(x, y_, threshold):
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
    return NetworkNodes(y, accuracy, prec, rec, train, variables, tf.merge_summary(summaries))

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
    variables = []
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
            variables += [w1, b1]
        #second convolutional layer
        with tf.name_scope('conv_layer2'):
            w2 = createWeights([5, 5, 32, 64])
            b2 = createBiases([64])
            c2 = createConv(p1, w2, b2)
            p2 = createPool(c2)
            #addSummaries(w1, b1, summaries, NET_NAME + '/conv_layer2')
            variables += [w2, b2]
        #densely connected layer
        with tf.name_scope('dense_layer'):
            w3 = createWeights([INPUT_HEIGHT//4 * INPUT_WIDTH//4 * 64, 1024])
            b3 = createBiases([1024])
            p2_flat = tf.reshape(p2, [-1, INPUT_HEIGHT//4 * INPUT_WIDTH//4 * 64])
            h1 = tf.nn.relu(tf.matmul(p2_flat, w3) + b3)
            #addSummaries(w3, b3, summaries, NET_NAME + '/dense_layer')
            variables += [w3, b3]
        #dropout
        h1_dropout = tf.nn.dropout(h1, p_dropout)
        #readout layer
        with tf.name_scope('readout_layer'):
            w4 = createWeights([1024, 2])
            b4 = createBiases([2])
            y  = tf.nn.softmax(tf.matmul(h1_dropout, w4) + b4)
            variables += [w4, b4]
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
            #add summaries
            summaries.append(tf.scalar_summary(NET_NAME + '/accuracy', accuracy))
            summaries.append(tf.scalar_summary(NET_NAME + '/precision', prec))
            summaries.append(tf.scalar_summary(NET_NAME + '/recall', rec))
    #return output nodes and trainer
    return NetworkNodes(y, accuracy, prec, rec, train, variables, tf.merge_summary(summaries))