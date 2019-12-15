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

sys.path.append('/home/max/Downloads/MTCNN/models/research/')
from PIL import Image
import cv2
from object_detection.utils import dataset_util # from path
from collections import namedtuple, OrderedDict # tf slim


def get_training_images(input_csv_list, input_image_list):
    files = []
    #input_image_list = '/home/max/Downloads/cosine_metric_learningMTcmt/MTCN/datasets/test_csv/'
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

def read_face_bbox(bbox_dir, output_dir):
    files = []
    for f in sorted(os.listdir(bbox_dir)):
        domain = os.path.abspath(bbox_dir)
        f = os.path.join(domain, f)
        files += [f]
        for line in open(f "r"):
            data =line.split(",")
            total_len = len(data)
            filename = data[0]
            bbox_info = [float[i] for i in data[2:total_len]]
            xmin = bbox_info[0]
            xmax = bbox_info[1]
            ymin = bbox_info[2]
            ymax = bbox_info[3]
            root = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/Val_dataset/'
            im_filename = os.path.join(root, filename)

            neg_xmin = 
            im = cv2.imread(im_filename)

