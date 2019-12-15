import os
import cv2
import random
import numpy as np 
from PIL import Image
import re

pos_csv_list = '/home/max/Desktop/train_data/pos_csv_list/'
neg_csv_list = '/home/max/Desktop/train_data/neg_csv_list/'
pos_image_list = '/home/max/Desktop/train_data/pos_samples/'
neg_image_list = '/home/max/Desktop/train_data/neg_samples/'

def get_pos_training_images(input_csv_list, input_image_list):
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
    print(files)


get_pos_training_images(pos_csv_list, pos_image_list)



# def get_annotations(input_csv_list, input_image_list, classes):

#     # Get the annotations of faces

#     files = get_training_images(input_csv_list, input_image_list)
#     # for i in image_list:
#     #     files.append(i)
#     labels = []
#     for i in range(len(files)):
#         face_to_int = class_text_to_int(classes)
#         labels.append(face_to_int)
#     annotation = [x for x in zip(files, labels)]
#     #annotation = zip(files, labels)
#     print(annotation)
#     return annotation
#     #annotation = zip(files, labels





# input_csv_list = '/home/max/Desktop/train_data/train_csv/'
# input_image_list = '/home/max/Desktop/train_data/train_dataset/'