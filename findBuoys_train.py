import os, time
import tensorflow as tf

from constants import *
from findBuoys_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from findBuoys_net import createCoarseNetwork, createDetailedNetwork

TRAINING_LOG_PERIOD  = 50 #informative lines are printed after this many training steps
TRAINING_SAVE_PERIOD = 1000 #save every N steps
TRAINING_RUN_PERIOD  = 50 #save runtime metadata every N steps

def train(dataFile, filterFile, useCoarseOnly, reinitialise, numSteps, threshold):
    startTime = time.time()
    #initialise
    cellFilter = getCellFilter(filterFile)
    if useCoarseOnly: #train coarse network
        net = createCoarseNetwork(threshold)
        prod = CoarseBatchProducer(dataFile, cellFilter)
        summaryWriter = tf.train.SummaryWriter(COARSE_SUMMARIES + "/train", net.graph)
        saveFile = COARSE_SAVE_FILE
    else: #train detailed network
        net = createDetailedNetwork()
        prod = DetailedBatchProducer(dataFile, cellFilter)
        summaryWriter = tf.train.SummaryWriter(DETAILED_SUMMARIES + "/train", net.graph)
        saveFile = DETAILED_SAVE_FILE
    #begin session
    with tf.Session(graph=net.graph) as sess:
        saver = tf.train.Saver(tf.all_variables())
        #reinitialise or load values
        if reinitialise or not os.path.exists(saveFile):
            sess.run(tf.initialize_all_variables())
        else:
            saver.restore(sess, saveFile)
        #start training
        for step in range(numSteps):
            inputs, outputs = prod.getBatch(BATCH_SIZE)
            feedDict = {net.x: inputs, net.y_: outputs, net.p_dropout: 0.5}
            if step > 0 and step % TRAINING_RUN_PERIOD == 0: #occasionally save runtime metadata
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run(
                    [net.summaries, net.train],
                    feed_dict=feedDict,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata
                )
                summaryWriter.add_run_metadata(run_metadata, "step%03d" % step)
            else:
                summary, _ = sess.run(
                    [net.summaries, net.train],
                    feed_dict=feedDict
                )
            summaryWriter.add_summary(summary, step) #write summary data for viewing with tensorboard
            #occasionally print step and accuracy
            if step % TRAINING_LOG_PERIOD == 0 or step == numSteps-1:
                feedDict = {net.x: inputs, net.y_: outputs, net.p_dropout: 1.0}
                acc, prec, rec = sess.run(
                    [net.accuracy, net.precision, net.recall],
                    feed_dict=feedDict
                )
                rps = (outputs.argmax(1) == 0).sum() / len(outputs) #num positive samples / num samples
                print(
                    "%7.2f secs - step %4d, accuracy %.2f, precision %.2f, recall %.2f, rps %.2f" %
                    (time.time() - startTime, step, acc, prec, rec, rps)
                )
            #occasionally save variable values
            if step > 0 and step % TRAINING_SAVE_PERIOD == 0:
                saver.save(sess, saveFile)
        saver.save(sess, saveFile)
        summaryWriter.close()
