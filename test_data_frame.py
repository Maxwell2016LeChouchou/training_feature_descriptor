
# import tensorflow as tf
# import numpy as np

# x = tf.ones([1, 5, 5, 3])

# filters = tf.ones([3, 3, 3, 64])

# output = tf.nn.conv2d(x, filters, strides=[1,1,1,1], padding='SAME')

# with tf.Session() as sess:
#     out = sess.run(output)

#     print(out.shape)
#     print(np.squeeze(out))


import pandas as pd 
import sys 
import os

sys.path.append('/home/max/Downloads/MTCNN/models/research/')
from PIL import Image
from object_detection.utils import dataset_util # from path
from collections import namedtuple, OrderedDict # tf slim
 

#csv_input = '/home/max/Desktop/data_frame.csv'

# def split(df, group):
#     print('df')
#     print(df)
#     data = namedtuple('data', ['filename', 'object'])
#     print('data')
#     print(data)
#     gb = df.groupby(group)
#     print('gb')
#     print(gb)
#     return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


# if __name__ == "__main__":
#     example = pd.read_csv(csv_input)
#     grouped = split(example, 'filename')
#     print('grouped')
#     print(grouped)

input_image_list = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv/'
label = ['face']

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
            #print(filename_path)
            files.append(filename_path)

    return files       


def get_annotations(input_image_list, classes):

    files = get_training_images(input_image_list)
    # for i in image_list:
    #     files.append(i)
    labels = []
    for i in range(len(files)):
        labels.append(classes)
    annotation = [x for x in zip(files, labels)]
    #annotation = zip(files, labels)
    print(annotation)
    return annotation
    #annotation = zip(files, labels)

if __name__ == "__main__":
    get_annotations(input_image_list, label)