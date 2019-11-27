import tensorflow as tf 
import os 
import numpy as np 
import model
from network import network_definition
import matplotlib.pyplot as plt 
import math

num_classes = 1   #WIDERFACE dataset
image_width = 255 
image_height = 255
batch_size = 100
capacity = 32203
max_step = 100000
learning_rate = 0.0001

def training():

    logs_train_dir = '/home/max/Downloads..../image_train_set...'

    tfrecords_file = '/home/max/Downloads.../.tfrecords'
    train_batch, train_label_batch = model.read_and_decode(tfrecords_file, batch_size=batch_size)
    train_batch = tf.cast(train_batch, dtype=tf.float32)
    train_label_batch = tf.cast(train_label_batch, dtype=tf.int64)

    train_logits = model.inference(train_batch, batch_size, n_classes)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary_merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try: 
        for step in np.arange(max_step):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 50 == 0:
                print('step %d, train loss = %.2f, train_accuracy = %.2f%%' %(step, training, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == max_step:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()




