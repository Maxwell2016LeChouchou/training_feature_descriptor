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
from skimage import io, transform, color, util

sys.path.append('/home/maxwell/Downloads/MTCNN/models/research/')
from PIL import Image
from object_detection.utils import dataset_util # from path
from collections import namedtuple, OrderedDict # tf slim


def class_text_to_int(row_label):
    if row_label == ['face']:
        return 1
    else:
        None

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

def get_training_images(input_csv_list, input_image_list):
    files = []
    #input_image_list = '/home/maxwell/Downloads/cosine_metric_learningMTcmt/MTCN/datasets/test_csv/'
    for filename in sorted(os.listdir(input_csv_list)):
        image_list_path = os.path.join(input_csv_list, filename)
        #print(image_list_path)

        for line in open(image_list_path, "r"):
            data = line.split(",")
            image = data[0]
            filename_path = os.path.join(input_image_list,image)
            #print(filename_path)
            files.append(filename_path)

    return files 

def convert_to_tfrecord(output_path, mode, anno):

    
    filename = os.path.join(output_path, mode + '.tfrecords')

    with tf.python_io.TFRecordWriter(filename) as writer:

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
                        'image/height': _int_feature(rows),
                        'image/width': _int_feature(cols),
                        'image/depth': _int_feature(depth),
                        'image/class/label': _int_feature(classes),
                        'image/encoded': _bytes_feature(img.astype(np.float32).tobytes())

                    }
                )
            )
            writer.write(example.SerializeToString())
     
def get_annotations(input_csv_list, input_image_list, classes):

    # Get the annotations of faces

    files = get_training_images(input_csv_list, input_image_list)
    # for i in image_list:
    #     files.append(i)
    labels = []
    for i in range(len(files)):
        face_to_int = class_text_to_int(classes)
        labels.append(face_to_int)
    annotation = [x for x in zip(files, labels)]
    #annotation = zip(files, labels)
    print(annotation)
    return annotation
    #annotation = zip(files, labels


def get_annotations_neg(input_csv_list, input_image_list, classes):
    
    #Get the annotation of non faces

    files = get_training_images()



def main(_):

    # Train_tfrecord
    input_csv_list_train = '/home/max/Desktop/train_data/train_csv/'
    input_image_list_train = '/home/max/Desktop/train_data/'
    output_path_train = '/home/maxwell/Downloads/MTCNN/training_feature_descriptor/tfrecord/'

    # Eval_tfrecord
    input_csv_list_eval = '/home/maxwell/Downloads//MTCNN/training_feature_descriptor/tfrecord/eval_csv/'
    input_image_list_eval = '/home/maxwell/Downloads/MTCNN/training_feature_descriptor/tfrecord/eval_dataset/'
    output_path_eval = '/home/maxwell/Downloads/MTCNN/training_feature_descriptor/tfrecord/'

    label_pos = ['face']
    label_neg = ['Non-face']
    mode_train = tf.estimator.ModeKeys.TRAIN
    mode_eval = tf.estimator.ModeKeys.EVAL

    annotation_train = get_annotations(input_csv_list_train, input_image_list_train, label_pos)
    annotation_eval = get_annotations(input_csv_list_eval, input_image_list_eval, label)

    convert_to_tfrecord(output_path_train, mode_train, annotation_train)
    convert_to_tfrecord(output_path_eval, mode_eval, annotation_eval)

if __name__ == '__main__':
    tf.app.run()










