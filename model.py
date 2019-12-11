from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import tensorflow as tf
import os
import random
from tqdm import tqdm 
import numpy as np  
import pandas as pd 
import sys

sys.path.append('/home/max/Downloads/MTCNN/models/research/')
from PIL import Image
from object_detection.utils import dataset_util # from path
from collections import namedtuple, OrderedDict # tf slim

output_path = '/home/max/Downloads/MTCNN/training_feature_descriptor/tfrecord'

input_image_list = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv/'

class_label = ['face']

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

def get_training_images(input_image_list):
    files = []
    #input_image_list = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv/'
    for filename in sorted(os.listdir(input_image_list)):
        image_list_path = os.path.join(input_image_list, filename)
        #print(image_list_path)

        for line in open(image_list_path, "r"):
            data = line.split(",")
            image = data[0]
            filename_path = os.path.join(input_image_list,image)
            print(filename_path)
            files.append(filename_path)

    return files 


def convert_to_tfrecord(output_path, anno):

    
    image_path = get_training_images(input_image_list)

    with tf.python_io.TFRecordWriter(image_path) as writer:

        for fnm, classes in anno:

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
                        'image/height': _int_feature(height),
                        'image/width': _int_feature(width),
                        'image/depth': _int_feature(depth),
                        'image/class/label': _int_feature(classes),
                        'image/encoded': _bytes_feature(img.astype(np.float32).tobytes())

                    }
                )
            )
            writer.write(example.SerializeToString())
     


def get_annotations(directory, classes):
    
    files = get_training_images(directory)

    labels = []
    for i in range(len(files)):
        labels.append(classes)
    
    annotation = [x for x in zip(files, labels)]

    return annotation

def main(_):

    annotation = get_annotations(input_image_list, class_label)

    convert_to_tfrecord(output_path, annotation)

if __name__ == '__main__':
    tf.app.run()










