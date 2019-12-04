
import tensorflow as tf
import numpy as np

x = tf.ones([1, 5, 5, 3])

filters = tf.ones([3, 3, 3, 64])

output = tf.nn.conv2d(x, filters, strides=[1,1,1,1], padding='SAME')

with tf.Session() as sess:
    out = sess.run(output)

    print(out.shape)
    print(np.squeeze(out))
