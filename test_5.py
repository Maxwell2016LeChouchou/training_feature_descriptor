import os
import cv2
import random
import numpy as np 
from PIL import Image


def read_face_bbox(bbox_dir, output_dir):
    files = []
    for f in sorted(os.listdir(bbox_dir)):
        domain = os.path.abspath(bbox_dir)
        f = os.path.join(domain, f)
        files += [f]
        for line in open(f, "r"):
            data =line.split(",")
            total_len = len(data)
            filename = data[0]
            bbox_a = [float(i) for i in data[2:total_len]]
            xmin = bbox_a[0]
            xmax = bbox_a[1]
            ymin = bbox_a[2]
            ymax = bbox_a[3]
            root = '/home/maxwell/Desktop/train_data/train_images/'
            im_filename = os.path.join(root, filename)
            im = cv2.imread(im_filename)
            #dimensions = im.shape

            cropped_neg_samples = im[int(ymin):int(ymax), int(xmin):int(xmax), :]
            save_to = os.path.join(output_dir, filename)       

            os.makedirs(os.path.split(save_to)[0], exist_ok=True)
            cv2.imwrite(save_to, cropped_neg_samples)
            print('Saving ', save_to)

read_face_bbox('/home/maxwell/Desktop/train_data/train_csv', '/home/maxwell/Desktop/train_data/train_pos_samples')