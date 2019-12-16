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

MODES = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]

def class_text_to_int(row_label):
    if row_label == ['face']:
        return 1
    else:
        return 0

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

    #assert mode in MODES, "wrong mode"
    
    filename = os.path.join(output_path, mode + '.tfrecords')

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

def get_annotations(pos_csv_list, neg_csv_list, pos_image_list, neg_image_list, label_pos, label_neg):

    # pos_csv_list = '/home/maxwell/Desktop/train_data/pos_csv_list/'
    # neg_csv_list = '/home/maxwell/Desktop/train_data/neg_csv_list/'
    # pos_image_list = '/home/maxwell/Desktop/train_data/pos_samples/'
    # neg_image_list = '/home/maxwell/Desktop/train_data/neg_samples/'

    # Get the annotations of faces
    files_pos = get_training_images(pos_csv_list, pos_image_list)
    files_neg = get_training_images(neg_csv_list, neg_image_list)

    files = []
    files.extend(files_pos)
    files.extend(files_neg)

    # for i in image_list:
    #     files.append(i)
    labels = []
    for i in range(len(files_pos)):
        face_to_int = class_text_to_int(label_pos)
        labels.append(face_to_int)
    for j in range(len(files_pos), len(files)):
        non_face_to_int = class_text_to_int(label_neg)
        labels.append(non_face_to_int)

    annotation = [x for x in zip(files, labels)]
    random.shuffle(annotation)
    #print(annotation)
    return annotation
    #annotation = zip(files, labels

def main(_):

    # Training set
    pos_csv_list_train = '/home/maxwell/Desktop/train_data/pos_csv_list/'
    neg_csv_list_train = '/home/maxwell/Desktop/train_data/neg_csv_list/'
    pos_image_list_train = '/home/maxwell/Desktop/train_data/pos_samples/'
    neg_image_list_train = '/home/maxwell/Desktop/train_data/neg_samples/'
    output_path_train = '/home/maxwell/Desktop/train_data/train_tfrecord/'

    # Eval set
    pos_csv_list_eval = '/home/maxwell/Desktop/eval_data/eval_pos_csv/'
    neg_csv_list_eval = '/home/maxwell/Desktop/eval_data/neg_csv_list/'
    pos_image_list_eval = '/home/maxwell/Desktop/eval_data/'
    neg_image_list_eval = '/home/maxwell/Desktop/eval_data/neg_samples/'
    output_path_eval = '/home/maxwell/Desktop/eval_data/eval_tfrecord/'

    label_pos = ['face']
    label_neg = ['non-face']

    train_anno = get_annotations(pos_csv_list_train, neg_csv_list_train, pos_image_list_train, neg_image_list_train, label_pos, label_neg)

    eval_anno = get_annotations(pos_csv_list_eval, neg_csv_list_eval, pos_image_list_eval, neg_image_list_eval, label_pos, label_neg)


    mode_train = tf.estimator.ModeKeys.TRAIN
    mode_eval = tf.estimator.ModeKeys.EVAL

    convert_to_tfrecord(output_path_train, mode_train, train_anno)
    convert_to_tfrecord(output_path_eval, mode_eval, eval_anno)

if __name__ == '__main__':
    tf.app.run()










