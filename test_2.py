"""
Created on Wed Jul 26 18:17:37 2017
@author: hjxu
"""

import os
import numpy as np
import tensorflow as tf
#import input_data
import model
import create_records as cr
 
N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001
 
 
def run_training1():
    
    # you need to change the directories to yours.
#    train_dir = '/home/hjxu/PycharmProjects/01_cats_vs_dogs/data/train/'
    logs_train_dir = '/home/hjxu/PycharmProjects/01_cats_vs_dogs/logs/recordstrain/'
#    
#    train, train_label = input_data.get_files(train_dir)
    tfrecords_file = '/home/hjxu/PycharmProjects/01_cats_vs_dogs/tfrecords/test.tfrecords'
    train_batch, train_label_batch = cr.read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
    train_batch = tf.cast(train_batch,dtype=tf.float32)
    train_label_batch = tf.cast(train_label_batch,dtype=tf.int64)
    
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)        
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
       
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
               
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
run_training1()



""""
First version of the ResNet
""""

def residual_block(x, out_channels, down_sample, projection=False):
    in_channels = x.get_shape().as_list()[3]
    if down_sample:
        x = max_pool(x)
 
    output = conv2d_with_batch_norm(x, [3, 3, in_channels, out_channels], 1)
    output = conv2d_with_batch_norm(output, [3, 3, out_channels, out_channels], 1)
 
    if in_channels != out_channels:
        if projection:
            # projection shortcut
            input_ = conv2d(x, [1, 1, in_channels, out_channels], 2)
        else:
            # zero-padding
            input_ = tf.pad(x, [[0,0], [0,0], [0,0], [0, out_channels - in_channels]])
    else:
 
        input_ = x
 
    return output + input_
 
def residual_group(name,x,num_block,out_channels):
 
    assert num_block>=1,'num_block must greater than 1'
 
    with tf.variable_scope('%s_head'%name):
        output = residual_block(x, out_channels, True)
 
    for i in range (num_block-1):
        with tf.variable_scope('%s_%d' % (name,i+1)):
            output = residual_block(output,out_channels, False)
 



""""
Another version in Keras: ResNet-18
""""

from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
from keras import backend as K
 
 
def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    """
    conv2d -> batch normalization -> relu activation
    """
    x = Conv2D(nb_filter, kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
 
 
def shortcut(input, residual):
    """
    shortcut连接，也就是identity mapping部分。
    """
 
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
 
    identity = input
    # 如果维度不同，则使用1x1卷积进行调整
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3],
                           kernel_size=(1, 1),
                           strides=(stride_width, stride_height),
                           padding="valid",
                           kernel_regularizer=regularizers.l2(0.0001))(input)
 
    return add([identity, residual])
 
 
def basic_block(nb_filter, strides=(1, 1)):
    """
    基本的ResNet building block，适用于ResNet-18和ResNet-34.
    """
    def f(input):
 
        conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
        residual = conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))
 
        return shortcut(input, residual)
 
    return f
 
 
def residual_block(nb_filter, repetitions, is_first_layer=False):
    """
    构建每层的residual模块，对应论文参数统计表中的conv2_x -> conv5_x
    """
    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2)
            input = basic_block(nb_filter, strides)(input)
        return input
 
    return f
 
 
def resnet_18(input_shape=(224,224,3), nclass=1000):
    """
    build resnet-18 model using keras with TensorFlow backend.
    :param input_shape: input shape of network, default as (224,224,3)
    :param nclass: numbers of class(output shape of network), default as 1000
    :return: resnet-18 model
    """
    input_ = Input(shape=input_shape)
 
    conv1 = conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
 
    conv2 = residual_block(64, 2, is_first_layer=True)(pool1)
    conv3 = residual_block(128, 2, is_first_layer=True)(conv2)
    conv4 = residual_block(256, 2, is_first_layer=True)(conv3)
    conv5 = residual_block(512, 2, is_first_layer=True)(conv4)
 
    pool2 = GlobalAvgPool2D()(conv5)
    output_ = Dense(nclass, activation='softmax')(pool2)
 
    model = Model(inputs=input_, outputs=output_)
    model.summary()
 
    return model
 
if __name__ == '__main__':
    model = resnet_18()
    plot_model(model, 'ResNet-18.png')  # 保存模型图

