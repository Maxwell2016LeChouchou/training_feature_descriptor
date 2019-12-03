import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim 


def cls_loss(logits, lables):
    """
    compute loss from train and labels
    Args: 
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
    Returns:
        loss tensor of float type
    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss



def cosine_dis_loss(a, b):
    """
    Compute element-wise cosine distance between `a` and `b`.

    Parameters
    ----------
    a : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    b : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.

    Returns
    -------
    tf.Tensor
        A matrix of shape NxM where element (i, j) contains the cosine distance
        between elements `a[i]` and `b[j]`.

    """

    a_normed = tf.nn.l2_normalize(a, dim=1)
    b_normed = a_normed if b is None else tf.nn.l2_normalize(b, dim=1)
    return (
        tf.constant(1.0, tf.float32) -
        tf.matmul(a_normed, tf.transpose(b_normed)))      

 



