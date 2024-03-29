
# train_dataset_size = 32203
# #train_batch_size = 1000
# image_size = 255
# #class = 1 #Widerface 

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


def inference(images, batch_size, n_classes):
    """
    Build the model
    Args: 
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    """    

    # conv1, shape = [kernel size, kernel size, channels, kernal numbers]

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', shape= [3,3,3,16], dtype = tf.float32, 
                                    initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[16], dtype=tf.float32, 
                                    initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activate, name=scope.name)

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                                padding = 'SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', shape=[3,3,16,16], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[16], dtype=tf.float32, 
                                    initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    #pool2 and norm2
    with tf.variable.scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME', name='pooling2')

    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_variable()[1].value
        weights = tf.get_variable('weight', shape=[dim,128], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', shape=[128, 128], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear', shape=[128, n_classes], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


def losses(logits, labels):

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits = logits, labels=labels, name='entropy_per_example')
        
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






