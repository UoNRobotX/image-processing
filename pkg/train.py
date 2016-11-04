import os, re, time
import tensorflow as tf

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .networks import createCoarseNetwork, createDetailedNetwork

TRAINING_LOG_PERIOD  = 50 #informative lines are printed after this many training steps
TRAINING_SAVE_PERIOD = 1000 #save every N steps
TRAINING_RUN_PERIOD  = 50 #save runtime metadata every N steps

def train(dataFile, dataFile2, filterFile, useCoarseOnly, reinitialise, outFile, numSteps, threshold):
    startTime = time.time()
    #initialise
    cellFilter = getCellFilter(filterFile)
    if useCoarseOnly: #train coarse network
        net = createCoarseNetwork(threshold)
        saveFile = COARSE_SAVE_FILE
        prod = CoarseBatchProducer(dataFile, cellFilter, outFile and outFile + "_train")
        testProd = CoarseBatchProducer(dataFile2, cellFilter, outFile and outFile + "_test")
        summaryWriter = tf.train.SummaryWriter(COARSE_SUMMARIES + "/train", net.graph)
        testSummaryWriter = tf.train.SummaryWriter(COARSE_SUMMARIES + "/train_test", net.graph)
    else: #train detailed network
        net = createDetailedNetwork()
        saveFile = DETAILED_SAVE_FILE
        prod = DetailedBatchProducer(dataFile, cellFilter, outFile and outFile + "_train")
        testProd = DetailedBatchProducer(dataFile2, cellFilter, outFile and outFile + "_test")
        summaryWriter = tf.train.SummaryWriter(DETAILED_SUMMARIES + "/train", net.graph)
        testSummaryWriter = tf.train.SummaryWriter(DETAILED_SUMMARIES + "/train_test", net.graph)
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
            if step > 0 and step % TRAINING_RUN_PERIOD == 0: #occasionally save runtime metadata
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run(
                    [net.summaries, net.train],
                    feed_dict={net.x: inputs, net.y_: outputs, net.p_dropout: 0.5},
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata
                )
                summaryWriter.add_run_metadata(run_metadata, "step%03d" % step)
            else:
                summary, _ = sess.run(
                    [net.summaries, net.train],
                    feed_dict={net.x: inputs, net.y_: outputs, net.p_dropout: 0.5}
                )
            summaryWriter.add_summary(summary, step) #write summary data for viewing with tensorboard
            #occasionally print step and accuracy
            if step % TRAINING_LOG_PERIOD == 0 or step == numSteps-1:
                inputs, outputs = testProd.getBatch(BATCH_SIZE)
                acc, prec, rec = sess.run(
                    [net.accuracy, net.precision, net.recall],
                    feed_dict={net.x: inputs, net.y_: outputs, net.p_dropout: 1.0}
                )
                testSummaryWriter.add_summary(summary, step)
                rps = (outputs.argmax(1) == 0).sum() / len(outputs) #num positive samples / num samples
                print(
                    "%7.2f secs - step %4d, accuracy %.2f, precision %.2f, recall %.2f, rps %f" %
                    (time.time() - startTime, step, acc, prec, rec, rps)
                )
            #occasionally save variable values
            if step > 0 and step % TRAINING_SAVE_PERIOD == 0:
                saver.save(sess, saveFile)
        saver.save(sess, saveFile)
        summaryWriter.close()
