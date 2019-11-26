import tensorflow as tf 
import os 
import numpy as np 
import model
from network import network_definition

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

    train_logits = network_definition.create_network(batch_size, num_classes=None, add_logits=True, reuse=None,
                   create_summaries=True, weight_decay=1e-8)



