import tensorflow as tf 

train_dataset_size = 32203
#train_batch_size = 1000
image_size = 255
#class = 1 #Widerface 

def read_and_decode(tfrecords_file, batch_size):
    ''''
    read and decode tf_records file of WIDERFACE, and generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns: 
        images: 4D tensor --[batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    ''''
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img,[image_size, image_size, 1])
    label = tf.cast(img_features['label'], tf.float32)
    img = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_batch = batch_size,
                                                num_threads = 64,
                                                capacity=train_dataset_size)
                                                
    return image_batch, tf.reshape(label_batch, [batch_size])
    print("finished decoding!!!!!!!!!!!!")


def losses(logits, labels):

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits = logits, labels=labels, name='xentropy_per_example')
        
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def training(loss, learning_rate):


    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, lables):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy






