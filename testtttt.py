
# import tensorflow as tf
# import numpy as np

# x = tf.ones([1, 5, 5, 3])

# filters = tf.ones([3, 3, 3, 64])

# output = tf.nn.conv2d(x, filters, strides=[1,1,1,1], padding='SAME')

# with tf.Session() as sess:
#     out = sess.run(output)

#     print(out.shape)
#     print(np.squeeze(out))


import os

def training_images(input_image_list):
    #input_image_list = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv/'
    for filename in sorted(os.listdir(input_image_list)):
        image_list_path = os.path.join(input_image_list, filename)
        #print(image_list_path)

        for line in open(image_list_path, "r"):
            data = line.split(",")
            image = data[0]
            filename_path = os.path.join(input_image_list,image)
            print(filename_path)

training_images('/home/max/Downloads/cosine_metric_learning/datasets/test_csv/')


