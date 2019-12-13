
# # import tensorflow as tf
# # import numpy as np

# # x = tf.ones([1, 5, 5, 3])

# # filters = tf.ones([3, 3, 3, 64])

# # output = tf.nn.conv2d(x, filters, strides=[1,1,1,1], padding='SAME')

# # with tf.Session() as sess:
# #     out = sess.run(output)

# #     print(out.shape)
# #     print(np.squeeze(out))


# import pandas as pd 
# import sys 
# import os

# sys.path.append('/home/max/Downloads/MTCNN/models/research/')
# from PIL import Image
# from object_detection.utils import dataset_util # from path
# from collections import namedtuple, OrderedDict # tf slim
 

# #csv_input = '/home/max/Desktop/data_frame.csv'

# # def split(df, group):
# #     print('df')
# #     print(df)
# #     data = namedtuple('data', ['filename', 'object'])
# #     print('data')
# #     print(data)
# #     gb = df.groupby(group)
# #     print('gb')
# #     print(gb)
# #     return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


# # if __name__ == "__main__":
# #     example = pd.read_csv(csv_input)
# #     grouped = split(example, 'filename')
# #     print('grouped')
# #     print(grouped)


# def get_training_images(input_csv_list, input_image_list):
#     files = []
#     #input_image_list = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv/'
#     for filename in sorted(os.listdir(input_csv_list)):
#         image_list_path = os.path.join(input_csv_list, filename)
#         #print(image_list_path)

#         for line in open(image_list_path, "r"):
#             data = line.split(",")
#             image = data[0]
#             filename_path = os.path.join(input_image_list,image)
#             #print(filename_path)
#             files.append(filename_path)

#     return files       


# def get_annotations(input_csv_list, input_image_list, classes):

#     files = get_training_images(input_csv_list, input_image_list)
#     # for i in image_list:
#     #     files.append(i)
#     labels = []
#     for i in range(len(files)):
#         labels.append(classes)
#     annotation = [x for x in zip(files, labels)]
#     #annotation = zip(files, labels)
#     print(annotation)
#     return annotation
#     #annotation = zip(files, labels)

# if __name__ == "__main__":
#     input_csv_list = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv/'
#     input_image_list = '/home/max/Downloads/cosine_metric_learning/datasets/test_dataset/'
#     label = ['face']

#     get_annotations(input_csv_list, input_image_list, label)





# def convert_to_tfrecord(output_path, anno):
    
#     with tf.python_io.TFRecordWriter(output_path) as writer:

#         for fnm, classes in anno:

#             # read and convert
#             img = io.imread(fnm)
#             img = color.rgb2gray(img)
#             img = transform.resize(img, [224, 224])

#             if 3 == img.ndim:
#                 rows, cols, depth = img.shape
#             else:
#                 rows, cols = img.shape
#                 depth = 1
            
#             example = tf.train.Example(
#                 features=tf.train.Features(
#                     feature={
#                         'image/height': _int_feature(rows),
#                         'image/width': _int_feature(cols),
#                         'image/depth': _int_feature(depth),
#                         'image/class/label': _int_feature(classes),
#                         'image/encoded': _bytes_feature(img.astype(np.float32).tobytes())

#                     }
#                 )
#             )
#             writer.write(example.SerializeToString())
     

# -*- coding:utf-8 -*-
__author__ = 'Leo.Z'
 
import os
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import glob
import tensorflow as tf
 
# 指定图片数据路径
PATH = 'cifar-10'
# 指定输出TFRecord文件
OUT_DIR = 'tfrecord_file'
 
 
# 以下函数将一个value转换为与tf适配的格式
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
 
 
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
 
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
 
 
# 利用一张图片创建一个example
def make_example(image_string, label):
    # 如果我们想获取图片的高宽和深度，则将读取到的数据先解码为jpeg图片格式，然后用shape取值
    # image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        # 'height': _int64_feature(image_shape[0]),
        # 'width': _int64_feature(image_shape[1]),
        # 'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string)
    }
 
    exam = tf.train.Example(features=tf.train.Features(feature=feature))
    return exam
 
 
# 定义一个TFRecordWriter实例
write = tf.io.TFRecordWriter(OUT_DIR)
 
# 创建一个index:img_name的字典
index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cate_list = ['airplane', 'mobilephone', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img_dict = zip(index_list, cate_list)
 
# 循环所有类别的图片文件夹，生成tfrecord文件
for index, img_cate in img_dict:
    # 生成训练集每个图片的路径
    train_img_list = glob.glob(os.path.join(PATH, 'train/{}/*.jpg'.format(index)))
    # test_img_list = glob.glob(os.path.join(PATH, 'test/{}/*.jpg'.format(index)))
 
    print("start to make train file index = {}:".format(index))
    count = 0
    # 开始循环读取每张图片，然后连同label，一起打包到TFRecord文件中
    for per_img_dir in train_img_list:
 
        # 使用gfile.Gfile来读图片文件
        with tf.io.gfile.GFile(per_img_dir, 'rb') as per_img_fp:
            img_data = per_img_fp.read()
        # 为每一张图片+label创建一个example
        example = make_example(img_data, index)
 
        # 打印看看example具体内容结构
        if count == 0:
            for line in str(example).split('\n')[:15]:
                print(line)
 
        # 将每个example写入到write中
        write.write(example.SerializeToString())
        count += 1
        if count % 1000 == 0:
            print(count)
 
write.close()
