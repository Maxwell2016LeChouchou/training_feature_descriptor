from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import tensorflow as tf 
from network import network_definition

data_dir = '/home/max/Downloads/MTCNN/training_feature_descriptor/tfrecord/'

MODES = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]

def input_fn(mode, batch_size=1):
    
    def parser(serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )

        image = tf.decode_raw(features['image_raw'],tf.float32)

        label = tf.cast(features['label'], tf.int32)

        return image, label
    
    if mode in MODES:
        tfrecords_file = os.path.join(data_dir, mode + '.tfrecords')
    else: 
        raise ValueError("Unknown mode")

    assert tf.gfile.Exists(tfrecords_file), ('TFRrecords does not exit')

    dataset = tf.data.TFRecordDataset(tfrecords_file)

    dataset = dataset.map(parser, num_parallel_calls=1)

    dataset = dataset.batch(batch_size)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()

    images, labels = iterator.get_next()

    return images, labels

def my_model_fn(features, labels, mode):

    tf.summary.image('image', features)

    logit_features = network_definition.factory_fn(features)

    predictions = {
        'classes': tf.argmax(input=logit_features, axis=1),
        'probabilities': tf.nn.softmax(logit_features, name='softmax_tensor')
    }

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logit_features, scope='loss')
    tf.summary.scalar('train_loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'],
        name='accuracy')

    accuracy_topk = tf.metrics.mean(
        tf.nn.in_top_k(predictions['probabilities'], tf.argmax(labels, axis=1), 2),
        name='accuracy_topk')
    
    metrics = {
        'test_accuracy': accuracy,
        'test_accuracy_topk': accuracy_topk
    }

    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_topk', accuracy_topk[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )

def main(_):
    logging_hook = tf.train.LoggingTensorHook(
        every_n_iter=100,
        tensors={
            'accuracy': 'accuracy/value',
            'accuracy_topk': 'accuracy_topk/value',
            'loss': 'loss/value'
        },
    )

    model = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir='/home/max/Downloads/MTCNN/training_feature_descriptor/model_store'
    )

    for i in range(20):
        model.train(
            input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN, batch_size=16),
            steps= 1000,
            hooks=[logging_hook]
        ) 

        print("=" * 10, "Testing", "=" * 10)
        eval_results = model.evaluate(
            input_fn=lambda: input_fn(tf.estimator.ModeKeys.EVAL))
        print('Evaluation results:\n\t{}'.format(eval_results))
        print("=" * 30)

        #Evaluation parts

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

    
