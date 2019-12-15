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
            width_bbox = xmax - xmin
            print(width_bbox)
            height_bbox = ymax - ymin
            root = '/home/maxwell/Desktop/train_data/train_images/'
            im_filename = os.path.join(root, filename)
            im = cv2.imread(im_filename)
            #dimensions = im.shape
            height_image = im.shape[0]
            width_image = im.shape[1]
            print(width_image)

            #counter = 3
            iou = 1
            #for i in counter:
            while iou > 0.1:
                bbox_b = []
                xmin_neg = random.randrange(width_image-width_bbox)  # xmin of negative sample 
                xmax_neg = xmin_neg + width_bbox                    # xmax of negative sample
                ymin_neg = random.randrange(height_image-height_bbox) # ymin of negative sample
                ymax_neg = ymin_neg + height_bbox                  # ymax of negative sample
                bbox_b.extend([xmin_neg, xmax_neg, ymin_neg, ymax_neg])
                iou = bbox_IOU(bbox_a, bbox_b)
                cropped_neg_samples = im[int(bbox_b[2]):int(bbox_b[3]), int(bbox_b[0]):int(bbox_b[1]), :]
                
                save_to = os.path.join(output_dir, filename)

                # Move out following folders due to the folder has too many big face picture to get neg samples
                # Alex_Ferguson, Alexandra_Pelosi, Alexander_Payne

            os.makedirs(os.path.split(save_to)[0], exist_ok=True)
            cv2.imwrite(save_to, cropped_neg_samples)
            print('Saving ', save_to)

            
def bbox_IOU(bbox_a, bbox_b):
    # determin the (x,y) ---- coordinates of the intersection rectangle 
    x_a = max(bbox_a[0], bbox_b[0])
    x_b = min(bbox_a[1], bbox_b[1])
    y_a = max(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    # Compute the area of intersection rectangle 
    interArea = max(0, x_b - x_a +1)*max(0,y_b - y_a+1)

    # Compute the area of face bbox:
    bbox_face = (bbox_a[1]-bbox_a[0]+1)*(bbox_a[3]-bbox_a[2]+1)

    # Compute IOU:
    iou = interArea/bbox_face

    return iou


read_face_bbox('/home/maxwell/Desktop/train_data/train_csv', '/home/maxwell/Desktop/train_data/train_neg_samples_3')
    