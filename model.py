from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import tensorflow as tf
import os
import random
from tqdm import tqdm 
import numpy as np  


output_train_path = '/home/maxwell/Downloads/MTCNN/training_feature_descriptor/image_set'

if not os.path.exists(output_train_path):
    os.makedirs(output_train_path)


MODES = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def convert_to_tfrecord(mode, anno):

    assert mode in MODES, "wrong tfrecord"

    filename = os.path.join(FLAGS.save_dir, mode + '.tfrecords')

    with tf.python_io.TFRecordWriter(filename) as writer:
        for fnm, classes in tqdm(anno):

            # read and convert
            img = io.imread(fnm)
            img = color.rgb2gray(img)
            img = transform.resize(img, [224, 224])

            if 3 == img.ndim:
                rows, cols, depth = img.shape
            else:
                rows, cols = img.shape
                depth = 1
            
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/height': _int_feature(rows),
                        'image/width': _int_feature(cols),
                        'image/depth': _int_feature(depth),
                        'image/class/label': _int_feature(classes),
                        'image/encoded': _bytes_feature(img.astype(np.float32).tobytes())

                    }
                )
            )
            writer.write(example.SerializeToString())


def get_training_images(input_image_list):
    #input_image_list = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv/'
    for filename in sorted(os.listdir(input_image_list)):
        image_list_path = os.path.join(input_image_list, filename)
        #print(image_list_path)

        for line in open(image_list_path, "r"):
            data = line.split(",")
            image = data[0]
            filename_path = os.path.join(input_image_list,image)
            print(filename_path)


def get_annotations(directory, classes):
    files = []
    labels = []

    for 









