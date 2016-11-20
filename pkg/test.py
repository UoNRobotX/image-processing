import os, time
import tensorflow as tf

from .constants import *
from .network_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from .network import createCoarseNetwork, createDetailedNetwork

def test(dataFile, filterFile, useCoarseOnly, reinitialise, outFile, numSteps, threshold):
    TESTING_LOG_PERIOD = 50
    TESTING_RUN_PERIOD = 50
    #initialise
    startTime = time.time()
    cellFilter = getCellFilter(filterFile)
    if useCoarseOnly: #test coarse network
        net = createCoarseNetwork(tf.Graph(), threshold)
        prod = CoarseBatchProducer(dataFile, cellFilter, outFile)
        batchSize = COARSE_BATCH_SIZE
        summaryDir = COARSE_SUMMARIES + "/test"
        saveFile = COARSE_SAVE_FILE
    else: #test detailed network
        net = createDetailedNetwork(tf.Graph())
        prod = DetailedBatchProducer(dataFile, cellFilter, outFile)
        batchSize = DETAILED_BATCH_SIZE
        summaryDir = DETAILED_SUMMARIES + "/test"
        saveFile = DETAILED_SAVE_FILE
    print("Startup time: %.2f secs" % (time.time() - startTime))
    rpsStrs = ["%.4f" % rps for rps in prod.getRps()]
    rpsStr = "[" + ", ".join(rpsStrs) + "]"
    print("Testing set size, rps: %d, %s" % (prod.getDatasetSize(), rpsStr))
    #test
    startTime = time.time()
    summaryWriter = tf.train.SummaryWriter(summaryDir, net.graph)
    metrics = [] #[[accuracy, precision, recall], ...]
    with tf.Session(graph=net.graph) as sess:
        #reinitialise or load values
        if reinitialise or not os.path.exists(saveFile):
            sess.run(tf.initialize_all_variables())
        else:
            tf.train.Saver(tf.all_variables()).restore(sess, saveFile)
        #do testing
        for step in range(numSteps):
            inputs, outputs = prod.getBatch(batchSize)
            feedDict = {net.x: inputs, net.y_: outputs, net.p_dropout: 1.0}
            if step > 0 and step % TESTING_RUN_PERIOD == 0: #if saving runtime metadata
                run_metadata = tf.RunMetadata()
                summary, acc, prec, rec = sess.run(
                    [net.summaries, net.accuracy, net.precision, net.recall],
                    feed_dict=feedDict,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata
                )
                summaryWriter.add_run_metadata(run_metadata, "step%03d" % step)
            else:
                summary, acc, prec, rec = sess.run(
                    [net.summaries, net.accuracy, net.precision, net.recall],
                    feed_dict=feedDict
                )
            metrics.append([acc, prec, rec])
            summaryWriter.add_summary(summary, step)
            if step % TESTING_LOG_PERIOD == 0:
                print(
                    "%7.2f secs - step %4d, acc %4.2f, prec %4.2f, rec %4.2f" %
                    (time.time()-startTime, step, acc, prec, rec)
                )
    accs  = [m[0] for m in metrics]
    precs = [m[1] for m in metrics]
    recs  = [m[2] for m in metrics]
    print("Averages: accuracy %4.2f, precision %4.2f, recall %4.2f" % \
        (avg(accs), avg(precs), avg(recs)))
    summaryWriter.close()

def avg(nums):
    return sum(nums)/len(nums)
