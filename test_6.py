import os
import cv2
import random
import numpy as np 
from PIL import Image
import re

def add_extra_dir(input_dir, output_dir):
    array_dic = []
    # files = []
    # for f in sorted(os.listdir(bbox_dir)):
    #     domain = os.path.abspath(bbox_dir)
    #     f = os.path.join(domain, f)
    #     files += [f]
    for line in open(input_dir, "r"):
        data =line.split(",")
        filename = data[0]
        face_label = data[1]
        total_len = len(data)
        bbox = [float(i) for i in data[2:total_len]]
        xmin = bbox[0]
        xmax = bbox[1]
        ymin = bbox[2]
        ymax = bbox[3]
        new_filename = 'eval_pos_samples/' + filename 
        array_dic.append(np.array([new_filename, face_label, xmin, xmax, ymin, ymax]))
    a = np.array(array_dic)
    np.savetxt(output_dir, a, fmt="%s,%s,%s,%s,%s,%s")
    print(output_dir)

def main():
    input_path = '/home/max/Desktop/eval_data/eval_csv/'
    output_path = '/home/max/Desktop/eval_data/eval_pos_csv/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for filename in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path,filename)):
            add_extra_dir(input_path+filename, output_path+filename)

if __name__ == "__main__":
    main()
