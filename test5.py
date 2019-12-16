
from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import os
import numpy as np  
import sys
import random

def class_text_to_int(row_label):
    if row_label == ['face']:
        return 1
    else:
        return 0


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
    print(annotation)
    return annotation
    #annotation = zip(files, labels


pos_csv_list = '/home/maxwell/Desktop/train_data/pos_csv_list/'
neg_csv_list = '/home/maxwell/Desktop/train_data/neg_csv_list/'
pos_image_list = '/home/maxwell/Desktop/train_data/pos_samples/'
neg_image_list = '/home/maxwell/Desktop/train_data/neg_samples/'
label_pos = ['face']
label_neg = ['non-face']
get_annotations(pos_csv_list, neg_csv_list, pos_image_list, neg_image_list, label_pos, label_neg)