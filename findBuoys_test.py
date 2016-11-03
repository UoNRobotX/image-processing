import os, time
import tensorflow as tf

from constants import *
from findBuoys_input import getCellFilter, CoarseBatchProducer, DetailedBatchProducer
from findBuoys_net import createCoarseNetwork, createDetailedNetwork

TESTING_LOG_PERIOD = 10
TESTING_RUN_PERIOD = 10

def test(dataFile, filterFile, useCoarseOnly, reinitialise, numSteps, threshold):
    startTime = time.time()
    #initialise
    cellFilter = getCellFilter(filterFile)
    if useCoarseOnly: #test coarse network
        net = createCoarseNetwork(threshold)
        prod = CoarseBatchProducer(dataFile, cellFilter)
        summaryWriter = tf.train.SummaryWriter(COARSE_SUMMARIES + '/test', net.graph)
        saveFile = COARSE_SAVE_FILE
    else: #test detailed network
        net = createDetailedNetwork()
        prod = DetailedBatchProducer(dataFile, cellFilter)
        summaryWriter = tf.train.SummaryWriter(DETAILED_SUMMARIES + '/test', net.graph)
        saveFile = DETAILED_SAVE_FILE
    with tf.Session(graph=net.graph) as sess:
        #reinitialise or load values
        if reinitialise or not os.path.exists(saveFile):
            sess.run(tf.initialize_all_variables())
        else:
            tf.train.Saver(tf.all_variables()).restore(sess, saveFile)
        #start testing
        metrics = [] #[[accuracy, precision, recall], ...]
        for step in range(numSteps):
            inputs, outputs = prod.getBatch(BATCH_SIZE)
            feedDict = {net.x: inputs, net.y_: outputs, net.p_dropout: 1.0}
            if step > 0 and step % TESTING_RUN_PERIOD == 0: #if saving runtime metadata
                run_metadata = tf.RunMetadata()
                summary, acc, prec, rec = sess.run(
                    [net.summaries, net.accuracy, net.precision, net.recall],
                    feed_dict=feedDict,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata
                )
                summaryWriter.add_run_metadata(run_metadata, 'step%03d' % step)
            else:
                summary, acc, prec, rec = sess.run(
                    [net.summaries, net.accuracy, net.precision, net.recall],
                    feed_dict=feedDict
                )
            metrics.append([acc, prec, rec])
            summaryWriter.add_summary(summary, step)
            if step % TESTING_LOG_PERIOD == 0:
                print(
                    "%7.2f secs - step %4d, accuracy %.2f, precision %.2f, recall %.2f" %
                    (time.time()-startTime, step, acc, prec, rec)
                )
    accs  = [m[0] for m in metrics]
    precs = [m[1] for m in metrics]
    recs  = [m[2] for m in metrics]
    print(
        "Averages: accuracy %.2f, precision %.2f, recall %.2f" %
        (sum(accs)/len(accs), sum(precs)/len(precs), sum(recs)/len(recs))
    )
    summaryWriter.close()
