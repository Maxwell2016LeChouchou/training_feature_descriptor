import os
import cv2
import random
import numpy as np 
from PIL import Image


def read_face_bbox(bbox_dir):
    files = []
    for f in sorted(os.listdir(bbox_dir)):
        domain = os.path.abspath(bbox_dir)
        f = os.path.join(domain, f)
        files += [f]
        for line in open(f, "r"):
            data =line.split(",")
            filename = data[0]
            new_file_path = 'test/' + filename 
            print(new_file_path)


read_face_bbox('/home/maxwell/Desktop/train_data/train_csv')