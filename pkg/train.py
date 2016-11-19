import os, re, time
import tensorflow as tf

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .network import createCoarseNetwork, createDetailedNetwork

def train(dataFile, dataFile2, filterFile, useCoarseOnly, reinitialise, outFile, numSteps, threshold):
    TRAINING_LOG_PERIOD  = 50 #informative lines are printed after this many training steps
    TRAINING_SAVE_PERIOD = 1000 #save every N steps
    TRAINING_RUN_PERIOD  = 50 #save runtime metadata every N steps
    #initialise
    startTime = time.time()
    cellFilter = getCellFilter(filterFile)
    if useCoarseOnly: #train coarse network
        net = createCoarseNetwork(tf.Graph(), threshold)
        prod = CoarseBatchProducer(dataFile, cellFilter, outFile and outFile + "_train")
        valProd = CoarseBatchProducer(dataFile2, cellFilter, outFile and outFile + "_validate")
        batchSize = COARSE_BATCH_SIZE
        summaryDir = COARSE_SUMMARIES + "/train"
        testSummaryDir = COARSE_SUMMARIES + "/validate"
        saveFile = COARSE_SAVE_FILE
    else: #train detailed network
        net = createDetailedNetwork(tf.Graph())
        prod = DetailedBatchProducer(dataFile, cellFilter, outFile and outFile + "_train")
        valProd = DetailedBatchProducer(dataFile2, cellFilter, outFile and outFile + "_validate")
        batchSize = DETAILED_BATCH_SIZE
        summaryDir = DETAILED_SUMMARIES + "/train"
        testSummaryDir = DETAILED_SUMMARIES + "/validate"
        saveFile = DETAILED_SAVE_FILE
    print("Startup time: %.2f secs" % (time.time() - startTime))
    print("Training set size and rps: %d, %.2f" % (prod.getDatasetSize(), prod.getRps()))
    print("Validation set size and rps: %d, %.2f" % (valProd.getDatasetSize(), prod.getRps()))
    #train
    startTime = time.time()
    summaryWriter = tf.train.SummaryWriter(summaryDir, net.graph)
    valSummaryWriter = tf.train.SummaryWriter(testSummaryDir, net.graph)
    with tf.Session(graph=net.graph) as sess:
        saver = tf.train.Saver(tf.all_variables())
        #reinitialise or load values
        if reinitialise or not os.path.exists(saveFile):
            sess.run(tf.initialize_all_variables())
        else:
            saver.restore(sess, saveFile)
        #do training
        p_dropout = TRAIN_DROPOUT
        prevAcc = 0.0
        for step in range(numSteps):
            inputs, outputs = prod.getBatch(batchSize)
            if step > 0 and step % TRAINING_RUN_PERIOD == 0: #occasionally save runtime metadata
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run(
                    [net.summaries, net.train],
                    feed_dict={net.x: inputs, net.y_: outputs, net.p_dropout: p_dropout},
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata
                )
                summaryWriter.add_run_metadata(run_metadata, "step%03d" % step)
            else:
                summary, _ = sess.run(
                    [net.summaries, net.train],
                    feed_dict={net.x: inputs, net.y_: outputs, net.p_dropout: p_dropout}
                )
            summaryWriter.add_summary(summary, step)
            #occasionally print step and accuracy
            if step % TRAINING_LOG_PERIOD == 0 or step == numSteps-1:
                inputs, outputs = valProd.getBatch(batchSize)
                acc, prec, rec = sess.run(
                    [net.accuracy, net.precision, net.recall],
                    feed_dict={net.x: inputs, net.y_: outputs, net.p_dropout: 1.0}
                )
                valSummaryWriter.add_summary(summary, step)
                rps = (outputs.argmax(1) == 0).sum() / len(outputs)
                    #num positive samples / num samples
                print(
                    "%7.2f secs - step %4d, acc %.2f (%+.2f), prec %.2f, rec %.2f, rps %.2f" %
                    (time.time() - startTime, step, acc, acc-prevAcc, prec, rec, rps)
                )
                prevAcc = acc
            #occasionally save variable values
            if step > 0 and step % TRAINING_SAVE_PERIOD == 0:
                saver.save(sess, saveFile)
        saver.save(sess, saveFile)
        summaryWriter.close()
